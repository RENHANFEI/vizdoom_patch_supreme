import re
import os
import json
import random
import inspect
import argparse
import subprocess
import torch
from torch import optim


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(string):
    """
    Parse boolean arguments from the command line.
    """
    if string.lower() in FALSY_STRINGS:
        return False
    elif string.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. "
                                         "use 0 or 1")


def map_ids_flag(string):
    """
    Parse map IDs.
    """
    ids = string.split(',')
    assert len(ids) >= 1 and all(x.isdigit() for x in ids)
    ids = sorted([int(x) for x in ids])
    assert all(x >= 1 for x in ids) and len(ids) == len(set(ids))
    return ids


def bcast_json_list(param, length):
    """
    Broadcast an parameter into a repeated list, unless it's already a list.
    """
    obj = json.loads(param)
    if isinstance(obj, list):
        assert len(obj) == length
        return obj
    else:
        assert isinstance(obj, int)
        return [obj] * length


def set_num_threads(n):
    """
    Set the number of CPU threads.
    """
    assert n >= 1
    torch.set_num_threads(n)
    os.environ['MKL_NUM_THREADS'] = str(n)


def get_device_mapping(gpu_id):
    """
    Reload models to the associated device.
    """
    origins = ['cpu'] + ['cuda:%i' % i for i in range(8)]
    target = 'cpu' if gpu_id < 0 else 'cuda:%i' % gpu_id
    return {k: target for k in origins}

def get_dump_path(main_dump_path, exp_name):
    """
        Create a directory to store the experiment.
        """
    assert len(exp_name) > 0
    # create the sweep path if it does not exist
    if not os.path.isdir(main_dump_path):
        subprocess.Popen("mkdir %s" % main_dump_path, shell=True).wait()
    sweep_path = os.path.join(main_dump_path, exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir %s" % sweep_path, shell=True).wait()
    # randomly generate a experiment ID
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        folder_name = ''.join(random.choice(chars) for _ in range(10))
        dump_path = os.path.join(sweep_path, folder_name)
        if not os.path.isdir(dump_path):
            break
    # create the dump folder
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir %s" % dump_path, shell=True).wait()
    return dump_path
