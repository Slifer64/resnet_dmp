import torchvision.utils
import cv2
from typing import Dict, List, Union
import copy
import os
import torchvision.transforms as torchvision_T
import my_pkg.data_transforms as data_tf
from my_pkg.data_types import *
from my_pkg.util.gmp import MovementPrimitive

import torch
import torchvision
import numpy as np
import pickle
import abc
from typing import List

__all__ = [
    "ResNetDMP",
    "VIMEDNet",
    "load_model",
    "mp_weights_to_traj",
    "mp_weights_trajectory_loss",
]


_tf_params = {
    'crop': 480,
    'resize': 224,
    'mean': np.array([0.0, 0.0, 0.0]),  # np.array([0.485, 0.456, 0.406])
    'std': np.array([1.0, 1.0, 1.0]),  # np.array([0.229, 0.224, 0.225])
    'fill': 0.9,
}
_tf_params['pad'] = (640 - _tf_params['crop']) / 2


def mp_weights_to_traj(W: torch.Tensor, duration: float, time_step: float, img_size=[1., 1.]) -> torch.Tensor:
    
    device = W.device
    dtype = W.dtype
    n_dofs = W.shape[1]
    n_kernels = W.shape[2]

    mp = MovementPrimitive(n_dofs=1, n_kernels=n_kernels)
    s_data = np.linspace(0, 1, int(duration / time_step + 0.5))
    Phi_data = torch.hstack([torch.tensor(mp.regress_vec(s)) for s in s_data]).to(device=device, dtype=dtype)

    # traj = torch.matmul(W.view(-1, n_kernels), Phi_data).view(W.shape[0], n_dofs, len(s_data)) # (batch_size, n_dofs, n_points)
    traj = torch.matmul(W, Phi_data) # (batch_size, n_dofs, n_points)

    h, w = img_size
    if h != 1. or w != 1.:
        traj = traj * torch.tensor([w, h], dtype=dtype , device=device).view(1, 2, 1)

    return traj


def mp_weights_trajectory_loss(pred_w: torch.Tensor, target_w: torch.Tensor, criterion=torch.nn.SmoothL1Loss, n_points=40, img_size=[1., 1.]) -> torch.Tensor:

    duration = 1.0
    time_step = duration / n_points

    y_pred = mp_weights_to_traj(pred_w, duration, time_step, img_size)
    y_target = mp_weights_to_traj(target_w, duration, time_step, img_size)

    # smoot_L1_loss = lambda x, beta, reduction=torch.mean: reduction(torch.where(x < beta**2, 0.5*x/beta, torch.sqrt(x) - 0.5*beta))
    # points_square_dist = lambda y_pred, y_target: torch.sum((y_pred - y_target)**2, dim=1)
    # return smoot_L1_loss(points_square_dist(y_pred, y_target), beta=0.2, reduction=torch.mean)

    return criterion(reduction='mean', beta=0.5)(y_pred, y_target)

# =======================================
# ============  Base Model  =============
# =======================================


class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._config = {}
        self.train_history = []

    def config_state_dict(self):
        return {'config': self._config, 'state_dict': self.state_dict(),
                'class_name': self.__class__.__name__,
                'train_history': self.train_history}

    @classmethod
    def from_config_state_dict(cls, s):
        class_name = s.pop('class_name', None)
        if not class_name:
            raise RuntimeError('Failed to load class name...')
        if class_name != cls.__name__:
            raise RuntimeError(f"Loaded class {class_name} != from called class {cls.__name__}")

        model = cls(**s['config'])
        model.load_state_dict(s['state_dict'])
        model.train_history = s['train_history']
        return model

    def save(self, filename: str):
        path = '/'.join(filename.split('/')[:-1])
        if path and not os.path.exists(path): os.makedirs(path)
        pickle.dump(self.config_state_dict(), open(filename, 'wb'))

    @classmethod
    def load(cls, filename):
        return cls.from_config_state_dict(pickle.load(open(filename, 'rb')))

    def _init_config(self, locals_dict):
        self._config = locals_dict
        self._config.pop('self')
        self._config.pop('__class__')

    @staticmethod
    @abc.abstractmethod
    def output_to_traj(net_output: Dict[str, torch.Tensor], duration=5.0, time_step=0.05, img_size=[1., 1.]) -> torch.Tensor:
        """
        Converts the networks output to a trajectory. The trajectory axes are scaled according to the img_size.

        Arguments:
        net_ouput -- torch.Tensor(batch_size, ...), the network's output.
        duration -- float, the time duration of the generated trajectory.
        time_step -- float, the time-step between consevutive trajectory points. 
        img_size -- Tuple[int, int] with the (height, width) of the image. (default=[1., 1.], i.e. no scaling)

        Returns:
        torch.Tensor(batch_size, n_dofs, n_points), the generated trajectory.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_function(pred, target, criterion):
        pass

    @staticmethod
    @abc.abstractmethod
    def input_transform(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def inv_input_transform(x):
        pass


# ======================================
# ============  ResNetDMP  =============
# ======================================

class ResNetDMP(BaseModel):

    global _tf_params

    __dataset_transforms = [
        data_tf.CenterCrop(_tf_params['crop']),
        data_tf.Resize(_tf_params['resize']),
        data_tf.Normalize(_tf_params['mean'], _tf_params['std'])
    ]

    __input_transforms = torchvision.transforms.Compose([
        torchvision_T.CenterCrop(_tf_params['crop']),
        torchvision_T.Resize(_tf_params['resize']),
        torchvision_T.Normalize(_tf_params['mean'], _tf_params['std'])
    ])

    __inv_input_transforms = torchvision.transforms.Compose([
        torchvision_T.Normalize(-_tf_params['mean'] / _tf_params['std'], 1 / _tf_params['std']),
        torchvision_T.Resize(_tf_params['crop']),
        torchvision_T.Pad([_tf_params['pad'], 0], fill=_tf_params['fill']),
    ])

    def __init__(self, mp_kernels: int, resnet_type='resnet18', in_channels=3, backbone_trainable=True):
        super().__init__()
        self._init_config(locals())

        self.mp_kernels = mp_kernels
        self.n_dofs = 2

        if resnet_type == 'resnet18':
            create_resnet = torchvision.models.resnet18
        elif resnet_type == 'resnet50':
            create_resnet = torchvision.models.resnet50
        else:
            raise RuntimeError(f'Unsupported resnet type "{resnet_type}"...')

        self.backbone = create_resnet(pretrained=False, progress=False)

        if in_channels != 3:
            c1 = self.backbone.conv1
            self.backbone.conv1 = torch.nn.Conv2d(in_channels, self.backbone.inplanes, kernel_size=c1.kernel_size,
                                                  stride=c1.stride, padding=c1.padding, bias=c1.bias)

        if not backbone_trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.bb_features = self.backbone.fc.in_features
        self.dmp_nn = torch.nn.Linear(self.bb_features, self.n_dofs * self.mp_kernels)
        self.backbone.fc = torch.nn.Flatten(start_dim=1)

        self.backbone_out_layers = {}
        self.backbone_out_layers['fc'] = 'dmp_features'

        from torchvision.models._utils import IntermediateLayerGetter
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=self.backbone_out_layers)

    @staticmethod
    def input_output_transforms():
        return ResNetDMP.__dataset_transforms

    @staticmethod
    def input_transform(x):
        return ResNetDMP.__input_transforms(x)

    @staticmethod
    def inv_input_transform(x):
        return ResNetDMP.__inv_input_transforms(x)

    @staticmethod
    def output_to_traj(net_output: Dict[str, torch.Tensor], duration=5.0, time_step=0.05, img_size=[1., 1.]) -> torch.Tensor:
        return mp_weights_to_traj(net_output['mp_weights'], duration, time_step, img_size)

    @staticmethod
    def loss_function(pred, target, criterion) -> torch.Tensor:
        return mp_weights_trajectory_loss(pred_w=pred['mp_weights'], target_w=target['mp_weights'], criterion=criterion)

    def forward(self, x):

        bb_out = self.backbone(x)

        out = {}
        out['mp_weights'] = self.dmp_nn(bb_out['dmp_features']).view(-1, self.n_dofs, self.mp_kernels)

        return out


# =====================================
# ============  VIMEDNet  =============
# =====================================


class VIMEDNet(BaseModel):

    global _tf_params

    __dataset_transforms = [
        data_tf.CenterCrop(_tf_params['crop']),
        data_tf.Resize(_tf_params['resize']),
        data_tf.Grayscale()
    ]

    __input_transforms = torchvision.transforms.Compose([
        torchvision_T.CenterCrop(_tf_params['crop']),
        torchvision_T.Resize(_tf_params['resize']),
        torchvision_T.Grayscale()
    ])

    __inv_input_transforms = torchvision.transforms.Compose([
        torchvision_T.Resize(_tf_params['crop']),
        torchvision_T.Pad([_tf_params['pad'], 0], fill=_tf_params['fill']),
        torchvision_T.Lambda(lambda x: x.repear(1, 3, 1, 1)),
    ])

    def __init__(self, mp_kernels):
        super().__init__()
        self._init_config(locals())

        self.mp_kernels = mp_kernels
        self.n_dofs = 2

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = torch.nn.Conv2d(in_channels=20, out_channels=35, kernel_size=30, stride=1, padding=14)

        self.global_max_pool = lambda x: torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:])

        # self.activ = torch.tanh
        self.activ = torch.nn.functional.relu

        self.fc1 = torch.nn.Linear(in_features=35, out_features=40)
        self.fc2 = torch.nn.Linear(in_features=40, out_features=45)
        self.fc3 = torch.nn.Linear(in_features=45, out_features=self.n_dofs * self.mp_kernels)

    @staticmethod
    def input_output_transforms():
        return VIMEDNet.__dataset_transforms

    @staticmethod
    def input_transform(x):
        return VIMEDNet.__input_transforms(x)

    @staticmethod
    def inv_input_transform(x):
        return VIMEDNet.__inv_input_transforms(x)

    @staticmethod
    def output_to_traj(net_output: Dict[str, torch.Tensor], duration=5.0, time_step=0.05, img_size=[1, 1]) -> torch.Tensor:
        return mp_weights_to_traj(net_output['mp_weights'], duration, time_step, img_size)

    @staticmethod
    def loss_function(pred, target, criterion) -> torch.Tensor:
        return mp_weights_trajectory_loss(pred_w=pred['mp_weights'], target_w=target['mp_weights'], criterion=criterion)

    def forward(self, x):

        x = self.activ(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activ(self.conv2(x))
        x = self.maxpool2(x)

        x = self.activ(self.conv3(x))
        x = self.maxpool3(x)

        x = self.activ(self.conv4(x))

        x = torch.flatten(self.global_max_pool(x), 1)

        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)

        return {'mp_weights': x.view(-1, self.n_dofs, self.mp_kernels)}


# ==========================================
# ==========================================


def load_model(filename: str):

    s = pickle.load(open(filename, 'rb'))

    class_name = s.pop('class_name', None)
    if not class_name:
        raise RuntimeError('Failed to load class name...')

    model_dict = {
        ResNetDMP.__name__: ResNetDMP,
        VIMEDNet.__name__: VIMEDNet,
    }

    if class_name not in model_dict:
        raise RuntimeError('Unsupported model type: "' + class_name + '"...')

    return model_dict[class_name].load(filename)