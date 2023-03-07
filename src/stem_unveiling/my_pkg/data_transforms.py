import cv2
import torch
import torchvision.transforms.functional as torch_F
import torchvision.transforms as torch_T
from typing import Optional, Dict, List
from my_pkg.data_types import *
from my_pkg.util.gmp import *
from typing import List, Union, Dict


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def add_transforms(self, transforms: List, i: int):
        self.transforms = self.transforms[:i] + transforms + self.transforms[i:]

    def __call__(self, inputs, outputs=None) -> Union[Dict, Tuple[Dict, Dict]]:
        for t in self.transforms:
            inputs, outputs = t(inputs, outputs)

        return (inputs, outputs) if outputs is not None else inputs


class Grayscale(torch_T.Grayscale):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:

        for name, input_ in inputs.items():
            inputs[name] = input_('grayscale')

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('grayscale')
        
        return (inputs, outputs) if outputs is not None else inputs


class GrayscaleToRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('grayscale_to_rgb')

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('grayscale_to_rgb')
        
        return (inputs, outputs) if outputs is not None else inputs


class CenterCrop(torch_T.CenterCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('center_crop', self.size)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('center_crop', self.size)
        
        return (inputs, outputs) if outputs is not None else inputs


class Resize(torch_T.Resize):
    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__(*args, **kwargs)

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('resize', self.size, self.interpolation)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('resize', self.size)
        
        return (inputs, outputs) if outputs is not None else inputs


class Normalize(torch_T.Normalize):
    def __init__(self, *args, **kwargs):
        super(Normalize, self).__init__(*args, **kwargs)

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('normalize', mean=self.mean, std=self.std)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('normalize', mean=self.mean, std=self.std)
        
        return (inputs, outputs) if outputs is not None else inputs


class Pad(torch_T.Pad):
    def __init__(self, *args, **kwargs):
        super(Pad, self).__init__(*args, **kwargs)

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('pad', padding=self.padding, fill=self.fill)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('pad', padding=self.padding, fill=self.fill)
        
        return (inputs, outputs) if outputs is not None else inputs


class Rotate(torch.nn.Module):
    def __init__(self, angle: float):
        super().__init__()
        self.angle = angle

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('rotate', self.angle)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('rotate', self.angle)
        
        return (inputs, outputs) if outputs is not None else inputs


class Translate(torch.nn.Module):
    def __init__(self, translate: List[int]):
        super().__init__()
        self.translate = translate

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('translate', self.translate)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('translate', self.translate)
        
        return (inputs, outputs) if outputs is not None else inputs


class Scale(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('scale', self.scale)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('scale', self.scale)
        
        return (inputs, outputs) if outputs is not None else inputs


class ToTensor(torch.nn.Module):
    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:

        for name, input_ in inputs.items():
            inputs[name] = input_('to_tensor')

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('to_tensor')
        
        return (inputs, outputs) if outputs is not None else inputs


class ToNumpy(torch.nn.Module):
    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('to_numpy')

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('to_numpy')
        
        return (inputs, outputs) if outputs is not None else inputs


class HorizontalFlip(torch.nn.Module):
    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('hflip')

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('hflip')
        
        return (inputs, outputs) if outputs is not None else inputs


class ColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        for name, input_ in inputs.items():
            inputs[name] = input_('color_jitter', brightness=self.brightness, contrast=self.contrast,
                                  saturation=self.saturation, hue=self.hue)

        if outputs is not None:
            for name, out in outputs.items():
                outputs[name] = out('color_jitter', brightness=self.brightness, contrast=self.contrast,
                                    saturation=self.saturation, hue=self.hue)
        
        return (inputs, outputs) if outputs is not None else inputs


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean, std, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:

        if torch.rand(1) < self.p:
            for name, input_ in inputs.items():
                inputs[name] = input_('gaussian_noise', mean=self.mean, std=self.std)
            if outputs is not None:
                for name, out in outputs.items():
                    outputs[name] = out('gaussian_noise', mean=self.mean, std=self.std)

        return (inputs, outputs) if outputs is not None else inputs

class RandomPerspective(torch.nn.Module):
    
    def __init__(self, distortion_scale=0.5, p=0.5):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.p = p

        self._mmr = {}

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None, repeat=False) -> Union[Dict, Tuple[Dict, Dict]]:

        if repeat and not self._mmr:
            p1 = self._mmr['p1']
            startpoints = self._mmr['startpoints']
            endpoints = self._mmr['endpoints']
        else:
            p1 = torch.rand(1)
            startpoints, endpoints = self._get_normalized_points(self.distortion_scale)
            self._mmr = {'p1': p1, 'startpoints': startpoints, 'endpoints': endpoints}

        if p1 < self.p:
            for name, input_ in inputs.items():
                inputs[name] = input_('perspective', startpoints, endpoints)
            if outputs is not None:
                for name, out in outputs.items():
                    outputs[name] = out('perspective', startpoints, endpoints)

        return (inputs, outputs) if outputs is not None else inputs

    @staticmethod
    def _get_normalized_points(distortion_scale: float) -> Tuple[List[List[int]], List[List[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        topleft = (torch.rand(2)*0.5*distortion_scale).tolist()
        topright = [1 - torch.rand(1).item()*0.5*distortion_scale, torch.rand(1).item()*0.5*distortion_scale] 
        botleft = [torch.rand(1).item()*0.5*distortion_scale, 1 - torch.rand(1).item()*0.5*distortion_scale] 
        botright = (1-torch.rand(2)*0.5*distortion_scale).tolist()
        startpoints = [[0., 0.], [1., 0.], [0., 1.], [1., 1.]]
        endpoints = [topleft, topright, botleft, botright]
        
        return startpoints, endpoints


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        if torch.rand(1) < self.p:
            inputs, outputs = HorizontalFlip()(inputs, outputs)

        return (inputs, outputs) if outputs is not None else inputs


class RandomRotate(torch.nn.Module):
    def __init__(self, angle_range: List[float]):
        super().__init__()
        self.angle_range = angle_range

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        angle = float(torch.empty(1).uniform_(self.angle_range[0], self.angle_range[1]).item())
        return Rotate(angle)(inputs, outputs)


class RandomTranslate(torch.nn.Module):
    def __init__(self, translate: Union[List[int], List[List[int]]]):
        super().__init__()
        if isinstance(translate[0], list):
            self.x_translate = translate[0]
            self.y_translate = translate[1]
        else:
            self.x_translate = [-translate[0], translate[0]] 
            self.y_translate = [-translate[1], translate[1]]

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        tx = int(round(torch.empty(1).uniform_(self.x_translate[0], self.x_translate[1]).item()))
        ty = int(round(torch.empty(1).uniform_(self.y_translate[0], self.y_translate[1]).item()))
        translate = [tx, ty]
        return Translate(translate)(inputs, outputs)


class RandomScale(torch.nn.Module):
    def __init__(self, scale_range: List[float]):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, inputs: Dict, outputs: Optional[Dict] = None) -> Union[Dict, Tuple[Dict, Dict]]:
        scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item())
        return Scale(scale)(inputs, outputs)


def check_transfom(inputs, outputs, transform):

    from my_pkg.util.cv_tools import draw_trajectory_on_image

    tf_image, tf_target = transform(inputs, outputs)
    tf_image = draw_trajectory_on_image(tf_image, tf_target.get_trajectory(), color=[255, 100, 255], linewidth=8)

    inputs = draw_trajectory_on_image(inputs.copy(), outputs.get_trajectory(), color=[255, 0, 0], linewidth=8)
    image2, _ = transform(inputs)

    comp_image = cv2.addWeighted(image2, 0.5, tf_image, 0.5, 0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].imshow(inputs)
    ax[0].set_title('Original')
    ax[1].imshow(tf_image)
    ax[1].set_title('Transformed')
    ax[2].imshow(comp_image)
    ax[2].set_title('Transformed vs Groundtruth')
    plt.pause(0.001)

