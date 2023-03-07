import os
import shutil

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def input_warning(msg: str):
    return input(bcolors.WARNING + bcolors.BOLD + msg + bcolors.ENDC)


def print_warning(msg: str):
    print(bcolors.WARNING + bcolors.BOLD + msg + bcolors.ENDC)


def print_error(msg: str):
    print('\033[91m' + '\033[1m' + msg + '\033[0m')


def print_warning(msg: str):
    print('\033[93m' + '\033[1m' + msg + '\033[0m')


def print_info(msg: str):
    print('\033[94m' + '\033[1m' + msg + '\033[0m')


def print_error(msg: str):
    print(bcolors.FAIL + bcolors.BOLD + msg + bcolors.ENDC)


def mkdir(path, reply=None):

    if os.path.exists(path):
        if reply is None:
            reply = input_warning('Path "' + path +
                                  '" already exists:\n- Remove [y]\n- Abort [n]\n- Proceed [p] \n> ').lower()
        if reply == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
        elif reply == 'p':
            pass
        else:
            print('Aborting...')
            exit()
    else:
        os.makedirs(path)


def mkdirs(path, reply=None):

    if os.path.exists(path):
        if reply is None:
            reply = input_warning('Path "' + path +
                                  '" already exists:\n- Remove [y]\n- Abort [n]\n- Proceed [p] \n> ').lower()
        if reply == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
        elif reply == 'p':
            pass
        else:
            print('Aborting...')
            exit()
    else:
        os.makedirs(path)
