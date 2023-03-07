import os
import argparse
import matplotlib.pyplot as plt
import torch

from my_pkg.dataset import StemUnveilDataset
from my_pkg.train_utils import train_model
from my_pkg.models import *
import my_pkg.data_transforms as data_tf

plt.ion()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_KERNELS = 10  # number of DMP kernels for DoF
OPTIMIZER = {'class': torch.optim.Adam, 'lr': 0.001}
LR_scheduler = {'gamma': 0.5, 'step_size': 40}
BASIC_LOSS = torch.nn.SmoothL1Loss

def parse_args():
    model_types = ('resnet18_dmp',
                   'resnet50_dmp',
                   'vimednet')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', required=True, type=str,
                        help='The type of the model.')
    parser.add_argument('--save_as', required=True, type=str,
                        help='The filename used to save the model.')
    parser.add_argument('--train_set', required=True, type=str,
                        help='The folder with the train dataset')
    parser.add_argument('--dev_set', type=str, default='',
                        help='The folder with the dev dataset (optional)')
    parser.add_argument('--test_set', type=str, default='',
                        help='The folder with the test dataset (optional)')            
    parser.add_argument('--epochs', default=200, type=int,
                        help='Training epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Training batch size.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for all random operations')

    args = vars(parser.parse_args())

    if args['model'] not in model_types:
        print('\33[33mModel should be one of the following: ' + str(model_types) + '\33[0m')
        exit()

    for set in ('train_set', 'dev_set', 'test_set'):
        if not os.path.exists(args[set]):
            print('\33[33mFolder "' + args[set] + '" does not exist...\33[0m')
            exit()

    return args


if __name__ == '__main__':

    args = parse_args()

    from my_pkg.util.rng import set_all_seeds
    set_all_seeds(args['seed'])

    # ========== Determine model type ===========
    model_type = args['model']
    resnet_type = model_type.split('_')[0]

    if model_type in ('resnet18_dmp', 'resnet50_dmp'):
        model = ResNetDMP(mp_kernels=N_KERNELS, resnet_type=resnet_type)
        train_fun = train_model
        kwargs = {'loss_fun': lambda y_hat, y: model.loss_function(y_hat, y, BASIC_LOSS)}
    elif model_type == 'vimednet':
        model = VIMEDNet(mp_kernels=N_KERNELS)
        train_fun = train_model
        kwargs = {'loss_fun': lambda y_hat, y: model.loss_function(y_hat, y, BASIC_LOSS)}
    else:
        raise RuntimeError("\33[1m\33[31mInvalid model type: %s \33[0m" % model_type)

    # ========== Load Datasets ===========
    data_loader = {'train_set': None, 'dev_set': None, 'test_set': None}
    for set in data_loader.keys():
        dataset = StemUnveilDataset.from_folder(args[set], data_transforms=data_tf.Compose(model.input_output_transforms()))
        data_loader[set] = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

    # ==========  Train the model  ============
    optimizer = OPTIMIZER['class'](model.parameters(), lr=OPTIMIZER['lr'])
    model = train_fun(
        model=model,
        optimizer=optimizer,
        **kwargs,
        epochs=args['epochs'],
        train_loader=data_loader['train_set'],
        val_loader=data_loader['dev_set'],
        test_loader=data_loader['test_set'],
        lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, gamma=LR_scheduler['gamma'],
                                                     step_size=LR_scheduler['step_size']),
        return_choice='best_val')

    # ==========  Plot loss  ============
    fig, axes = model.train_history.plot()

    # ==========  Save the model  ============
    model.save(args['save_as'])

    # input('Press [enter] to finish...')