import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import math
from typing import List, Union

import my_pkg.data_transforms as data_tf
from my_pkg.dataset import StemUnveilDataset
from my_pkg.models import *
from my_pkg.data_types import *
from my_pkg.util.cv_tools import draw_trajectory_on_image

plt.ion()

# for optional data-augmentation
data_augment_transforms = data_tf.Compose([
    data_tf.RandomHorizontalFlip(p=0.5),
    data_tf.RandomRotate([-12, 12]),
    data_tf.RandomScale([0.85, 1.15]),
    data_tf.RandomTranslate([15, 15]),
])

# color used to draw the trajectory of each model
demo_COLOR = (50, 255, 255)  # cyan
COLORS = [
    (255, 0, 0),     # red
    (0, 0, 255),   # blue
    (217, 84, 26),   # light brown
    (20, 200, 20),   # green
    (255, 0, 255),   # magenta
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models', required=True, type=str, nargs='+', 
                        help='The model filename to load')
    parser.add_argument('--dataset', required=True, type=str, 
                        help='The folder with the test dataset')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='How many samples to visualize each time')
    parser.add_argument('--shuffle', default=1, type=int, 
                        help='Whether to shuffle the dataset')
    parser.add_argument('--random_transforms', default=0, type=int,
                        help='Whether to apply random transforms to the dataset')
    parser.add_argument('--seed', default=0, type=int,  
                        help='Random seed.')

    args = vars(parser.parse_args())

    if not os.path.exists(args['dataset']):
        print('\33[33mDataset folder "' + args['dataset'] + '" does not exist...\33[0m')
        exit()

    for model in args['models']:
        if not os.path.exists(args['dataset']):
            print('f\33[33mModel "{model}" does not exist...\33[0m')
            exit()

    return args



def batch_imshow(inputs: Union[torch.Tensor, List[torch.Tensor]], demo_traj=None, pred_traj=None, figsize=(12, 9)):

    if isinstance(inputs, list):
        inputs = torch.stack(inputs, dim=0)

    images = torch.permute(inputs, [0, 2, 3, 1]).numpy()

    n_images = images.shape[0]
    if n_images < 4:
        n_rows = 1
        n_cols = n_images
    else:
        n_rows = int(math.sqrt(n_images) + 0.5)
        n_cols = int(n_images / n_rows + 0.5)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        ax = np.array([ax])
    if n_cols == 1:
        ax = ax[..., None]

    for row in ax:
        for ax_ in row:
            ax_.axis('off')
    for k in range(n_images):
        image = images[k]
        if demo_traj is not None:
            image = draw_trajectory_on_image(image, demo_traj[k], color=demo_COLOR, linewidth=7)
        for i, traj in enumerate(pred_traj):
            image = draw_trajectory_on_image(image, traj[k], color=COLORS[i], linewidth=6)
        i, j = np.unravel_index(k, (n_rows, n_cols))  # k // n_cols, k % n_cols
        ax[i, j].imshow(np.clip(image, 0, 1))
    plt.pause(0.001)

    return fig


if __name__ == '__main__':

    args = parse_args()

    from my_pkg.util.rng import set_all_seeds
    set_all_seeds(args['seed'])

    models = [load_model(model_f) for model_f in args['models']]
    for model in models: model.eval()

    # ========== Load Dataset ===========
    data_transforms = data_augment_transforms if args['random_transforms'] else None

    dataset = StemUnveilDataset.from_folder(args['dataset'], data_transforms=data_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                              shuffle=args['shuffle'], num_workers=4)

    # ========== Evaluate model ===========
    for i, model in enumerate(models):
        fig, axes = model.train_history.plot()
        # import scipy.io
        # save_data_as = args['model'][i].split('/')[-1].split('.')[0] + '_loss.mat'
        # scipy.io.savemat(save_data_as, model.train_history.history)

    for inputs, outputs in data_loader:
        rgb_img = inputs[InputImage.data_name]
        img_size = rgb_img.shape[2:4]
        demo_traj = mp_weights_to_traj(outputs['mp_weights'], duration=5.0, time_step=0.05, img_size=img_size)
        pred_traj = []
        with torch.no_grad():
            for model in models:
                out = model(model.input_transform(rgb_img))
                pred_traj.append(model.output_to_traj(out, img_size=img_size))
        fig = batch_imshow(rgb_img, demo_traj, pred_traj, figsize=(16, 13))
        input_ = input('Press [q] to exit or [enter] to continue').lower()
        if input_ == 'q':
            break

        plt.close(fig)



