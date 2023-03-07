import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import math

from my_pkg.dataset import StemUnveilDataset
from my_pkg.data_types import *
from my_pkg.util.cv_tools import draw_trajectory_on_image


INPUT_TYPES = (InputImage, )
OUTPUT_TYPES = (MpTrajectory, )


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, type=str, 
                        help='The folder with the dataset')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Number of data samples to vizualize at each step.')
    parser.add_argument('--shuffle', default=0, type=int, 
                        help='Whether to shuffle the data.')
    parser.add_argument('--seed', default=0, type=int, 
                        help='Rng seed')
    args = vars(parser.parse_args())
    
    return args


if __name__ == '__main__':

    plt.ion()

    args = parse_args()

    from my_pkg.util.rng import set_all_seeds
    set_all_seeds(args['seed'])

    dataset = StemUnveilDataset.from_folder(args['dataset'], in_types=INPUT_TYPES, out_types=OUTPUT_TYPES)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=args['shuffle'], num_workers=4)

    for inputs, outputs in data_loader:
        rgb_img = inputs[InputImage.data_name]
        mp_weights = outputs[MpTrajectory.data_name]

        n_cols = min(4, len(rgb_img)) 
        n_rows = int(len(rgb_img)/n_cols)

        batch_size = rgb_img.shape[0]
        if batch_size < 4:
            n_rows = 1
            n_cols = batch_size
        else:
            n_rows = int(math.sqrt(batch_size) + 0.5)
            n_cols = int(batch_size / n_rows + 0.5)
            

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(13, 8))
        if n_rows == 1: ax = np.array([ax])
        if n_cols == 1: ax = ax[..., None]
        for ax_row in ax:
            for ax_i in ax_row: ax_i.axis('off')

        for k in range(batch_size):
            img = rgb_img[k].numpy().transpose(1, 2, 0)
            traj = MpTrajectory(mp_weights[k]).get_trajectory(duration=1.0, time_step=0.05)
            img = draw_trajectory_on_image(img, traj, color=(255, 0, 0), linewidth=6)
            i, j = np.unravel_index(k, (n_rows, n_cols))  # k // n_cols, k % n_cols
            ax[i, j].imshow(img)
        plt.pause(0.001)
        print("To continue press [enter]. To stop type 's' and press [enter]")
        if input().lower() == 's': break
        plt.close(fig)
