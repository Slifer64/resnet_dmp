import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import torch
from typing import List, Optional, Dict, Tuple
import copy
import my_pkg.data_transforms as data_tf
from my_pkg.data_types import *


class StemUnveilDataset(torch.utils.data.Dataset):

    def __init__(self, data_samples: List[str], data_transforms=None, in_types=(InputImage, ), out_types=(MpTrajectory, ), opt_in_types=(), opt_out_types=()):
        if isinstance(data_transforms, list):
            data_transforms = data_tf.Compose(data_transforms)
        self.transforms = data_transforms
        self.samples = copy.deepcopy(data_samples)
        self.output_seg = False

        self.in_types = in_types
        self.out_types = out_types
        self.opt_in_types = opt_in_types
        self.opt_out_types = opt_out_types

    @classmethod
    def from_folder(cls, path: str, data_transforms=None, in_types=(InputImage, ), out_types=(MpTrajectory, ), opt_in_types=(), opt_out_types=()):
        # load all image files, sorting them to ensure that they are aligned
        data_samples = [os.path.join(path, i) for i in list(sorted(os.listdir(path)))
                        if os.path.isdir(os.path.join(path, i))]
        return cls(data_samples=data_samples, data_transforms=data_transforms, in_types=in_types, out_types=out_types, opt_in_types=opt_in_types, opt_out_types=opt_out_types)

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        batch_inputs, batch_outputs = zip(*[self.__call__(idx, return_raw=True) for idx in indices])
        return map(lambda x: {k: torch.stack([x_i[k] for x_i in x], dim=0) for k in x[0].keys()}, [batch_inputs, batch_outputs])

    def __call__(self, idx, return_raw=True, apply_tf=True):
        sample_path = self.samples[idx]

        # =========  parse mandutory inputs  =========
        inputs = {type_.data_name: type_.load(sample_path) for type_ in self.in_types}
        for k, v in inputs.items():
            if v is None:
                raise RuntimeError('\33[1;31m' + k + ' is empty!\33[0m')
                

        # =========  parse optional inputs  =========
        for type_ in self.opt_in_types:
            dat = type_.load(sample_path)
            if dat is not None:
                inputs[type_.data_name] = dat

        # =========  parse mandutory outputs  =========
        outputs = {type_.data_name: type_.load(sample_path) for type_ in self.out_types}

        # if there is no segmentation mask data, create an empty mask with the appropriate size.
        # This is essential to properly aggregate data in batches, where some data in the batch may have missing data,
        # so we fill them with empty data of the appropriate size.
        if SegmentationMask.data_name in outputs.keys() and outputs[SegmentationMask.data_name] is None:
            img_size = inputs[InputImage.data_name].size()[1:]
            outputs[SegmentationMask.data_name] = SegmentationMask.empty(img_size)

        for k, v in outputs.items():
            if v is None:
                raise RuntimeError('\33[1;31m' + k + ' is empty!\33[0m')

        # =========  parse optional outputs  =========
        for type_ in self.opt_out_types:
            dat = type_.load(sample_path)
            if dat is not None:
                outputs[type_.data_name] = dat

        # =========  apply transforms  =========
        if apply_tf and self.transforms is not None:
            inputs, outputs = self.transforms(inputs, outputs)

        # net_inputs = torch.concat([v.torch_tensor() for v in inputs.values()], dim=0)
        if return_raw:
            inputs = {name: input_.torch_tensor() for name, input_ in inputs.items()}
            outputs = {name: out.torch_tensor() for name, out in outputs.items()}

        return inputs, outputs

    def __getitem__(self, idx):
        return self(idx, return_raw=True)

    def __len__(self):
        return len(self.samples)

    def get_seed(self, idx):
        path = os.path.join(self.samples[idx], 'seed_*.txt')
        seed = int(glob.glob(path, recursive=False)[0].split('/')[-1].split('_')[-1].split('.')[0])
        return seed

    def subset(self, indices) -> 'StemUnveilDataset':
        return StemUnveilDataset(data_samples=[self.samples[i] for i in indices], data_transforms=self.transforms)
