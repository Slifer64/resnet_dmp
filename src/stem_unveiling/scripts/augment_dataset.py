import shutil
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import tqdm
import my_pkg.data_transforms as data_tf
from my_pkg.dataset import StemUnveilDataset
from my_pkg.data_types import *
from my_pkg.util.cv_tools import draw_trajectory_on_image

INPUT_TYPES = (InputImage, )
OPT_INPUT_TYPES = () # BeforeImage, AfterImage, SegImage)

OUTPUT_TYPES = (MpTrajectory, )
OPT_OUTPUT_TYPES = ()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iters', required=True, type=int, 
                        help='Number of times to iterate over each sample in the dataset')
    parser.add_argument('--dataset', required=True, type=str, 
                        help='The folder with the dataset')
    parser.add_argument('--save_to', required=True, type=str,
                        help='Where to save the augmented dataset')
    parser.add_argument('--transforms', required=True, type=str,
                        help='File with the data transforms')
    parser.add_argument('--seed', default=0, type=int, 
                        help='Rng seed')
    parser.add_argument('--append', default=0, type=int, 
                        help='Whether to append to existing folder or remove and re-create it.')
    parser.add_argument('--viz', default=0, type=int,
                        help='Flag to visualize or not the transformed data')
    args = vars(parser.parse_args())
    
    if not args['append'] and os.path.exists(args['save_to']):
        shutil.rmtree(args['save_to'])

    if not os.path.exists(args['save_to']):
        os.makedirs(args['save_to'])

    with open(args['transforms'], 'r') as fid:
        args['transforms'] = fid.read()

    return args


if __name__ == '__main__':

    plt.ion()

    args = parse_args()

    from my_pkg.util.rng import set_all_seeds
    set_all_seeds(args['seed'])

    data_augment_transforms = eval(args['transforms'])

    dataset = StemUnveilDataset.from_folder(args['dataset'], data_transforms=data_augment_transforms, in_types=INPUT_TYPES, out_types=OUTPUT_TYPES, opt_in_types=OPT_INPUT_TYPES, opt_out_types=OPT_OUTPUT_TYPES)

    count = len(next(os.walk(args['save_to']))[1]) if args['append'] else 0

    outer_iters = args['iters']
    for k in range(outer_iters):

        print(f'\33[1;34m*** outer iter: {k+1:2d}/{outer_iters}\33[0m')

        for i in tqdm.tqdm(range(len(dataset))):

            count += 1
            sample_path = os.path.join(args['save_to'], str(count).zfill(4))

            if os.path.exists(sample_path):
                shutil.rmtree(sample_path)
            os.mkdir(sample_path)

            # do not return raw types, so that we can call the 'save' function of the datatype afterwards
            inputs, outputs = dataset(i, return_raw=False)

            rgb0 = InputImage.load(dataset.samples[i]).numpy()
            traj = MpTrajectory.load(dataset.samples[i]).get_trajectory(duration=1.0, time_step=0.05, img_size=rgb0.shape[:2])
            rgb0 = draw_trajectory_on_image(rgb0, traj, color=(255, 0, 0), linewidth=6)
            
            for name, input_ in inputs.items():
                input_.save(sample_path)

            for name, output_ in outputs.items():
                output_.save(sample_path)

            # ======= To visualize the transformed data ========
            if args['viz']:
                rgb = inputs[InputImage.data_name].numpy()
                before_img = inputs[BeforeImage.data_name].numpy() if BeforeImage.data_name in inputs.keys() else np.zeros_like(rgb)
                after_img = inputs[AfterImage.data_name].numpy() if AfterImage.data_name in inputs.keys() else np.zeros_like(rgb) 
                seg_image = inputs[SegImage.data_name].numpy()  if SegImage.data_name in inputs.keys() else np.zeros_like(rgb)
                seg_mask = outputs[SegmentationMask.data_name].numpy()
                
                traj = outputs[MpTrajectory.data_name].get_trajectory(duration=1.0, time_step=0.05, img_size=rgb.shape[:2])
                rgb_traj = draw_trajectory_on_image(rgb, traj, color=(255, 0, 0), linewidth=6)

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.axis('off')
                im_viz = np.hstack([rgb0, np.ones((rgb.shape[0], 30, 3)) ,rgb_traj])
                ax.imshow(im_viz)
                plt.pause(0.0001)
                
                if input().lower() == 's':
                    break
                
                plt.close(fig)
