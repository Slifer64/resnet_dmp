import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from typing import Tuple
from my_pkg.dataset import StemUnveilDataset
from my_pkg.models import *
from my_pkg.data_types import *
from my_pkg.util.cv_tools import draw_trajectory_on_image, batch_imshow


plt.ion()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str,
                        help='The model\'s filename.')
    parser.add_argument('--datasets', required=True, type=str, nargs='+',
                        help='One or more dataset folders on which to calc the RMSE.')
    parser.add_argument('--plot_worst_n', default=0, type=int,
                        help='Plot the worst "n" samples.')

    args = vars(parser.parse_args())

    for dataset in args['datasets']:
        if not os.path.exists(dataset):
            print(f'\33[33mDataset folder "{dataset}" does not exist...\33[0m')
            exit()

    return args

    
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean(torch.sum((yhat - y)**2, dim=1), dim=1))


def calc_RMSE_loss(model, dataset: StemUnveilDataset, verbose=False, batch_size=32) -> Tuple[float, float, float]:

    model.to(device)
    model.eval()

    loss_data = []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    from tqdm import tqdm
    progress_bar = tqdm(total=len(dataset))

    for inputs, outputs in data_loader:
        # inputs, outputs = dataset[i]
        rgb_img = inputs[InputImage.data_name].to(device)
        mp_weights = outputs[MpTrajectory.data_name].to(device)
        img_size = rgb_img.shape[2:4] # (H,W)
        with torch.no_grad():
            output = model(model.input_transform(rgb_img))
        pred_traj = model.output_to_traj(output, duration=5, time_step=0.05, img_size=img_size)
        demo_traj = mp_weights_to_traj(mp_weights, duration=5, time_step=0.05, img_size=img_size)
        loss_i = RMSELoss(pred_traj, demo_traj).view(-1)
        loss_data = loss_data + loss_i.tolist()

        progress_bar.update(rgb_img.shape[0])

    progress_bar.close()

    loss_data = np.array(loss_data)
    mean_loss = np.mean(loss_data)
    std_loss = np.std(loss_data)
    max_loss = np.max(loss_data)
    if verbose:
        print(f'loss = {mean_loss:.2f} +/- {std_loss:.2f} px')
        print(f'max_loss = {max_loss:.2f} px')

    return mean_loss, std_loss, max_loss, loss_data



if __name__ == '__main__':

    args = parse_args()

    model = load_model(args['model'])
    model.eval()

    # print(f"\033[1;36m=====  Measuring model '{args['model']}' ======\033[0m")

    for dataset_name in args['datasets']:
        dataset = StemUnveilDataset.from_folder(dataset_name)
        mean_loss, std_loss, max_loss, loss_data = calc_RMSE_loss(model, dataset, batch_size=32)

        mean_loss, std_loss = mean_loss, std_loss

        print(f'=== {dataset_name} ====')
        print(f'loss = {mean_loss:.2f} +/- {std_loss:.2f} px')
        print(f'max_loss = {max_loss:.2f} px')

        if args['plot_worst_n'] > 0:
            ind = np.argsort(loss_data)[-args['plot_worst_n']:]
            inputs, outputs = dataset.get_batch(ind)
            rgb_img = inputs[InputImage.data_name].to(device)
            mp_weights = outputs[MpTrajectory.data_name].to(device)
            img_size = rgb_img.shape[2:4] # (H,W)
            with torch.no_grad():
                output = model(model.input_transform(rgb_img))
            pred_traj = model.output_to_traj(output, duration=5, time_step=0.05, img_size=img_size).detach().cpu().numpy()
            demo_traj = mp_weights_to_traj(mp_weights, duration=5, time_step=0.05, img_size=img_size).detach().cpu().numpy()

            rgb_img = rgb_img.permute(0, 2, 3, 1).detach().cpu().numpy() # restore channels order
            for i, img in enumerate(rgb_img):
                img = draw_trajectory_on_image(img, demo_traj[i], color=(50, 255, 255), linewidth=7)
                img = draw_trajectory_on_image(img, pred_traj[i], color=(255, 0, 0), linewidth=6)
                rgb_img[i] = img

            batch_imshow(rgb_img)
            print('loss_data = ', loss_data[ind])
            input('....')
                    
