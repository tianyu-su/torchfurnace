# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:03

"""
some functions for learning model
"""
import sys

__author__ = 'tianyu'

import os
import re
from pathlib import Path

import torch
from tqdm import tqdm

from .meter import AverageMeter


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    :param output: output scores shape=(N,cls_num)
    :param target: target label  shape=(N)
    :param topk: a tuple of expectation K
    :return: a list of topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return tuple(res)


def load_pretrained(model, pth):
    if os.path.isfile(pth):
        log(f'Loading pretrained <==> {pth}')
        checkpoint = torch.load(pth, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        # log("=> loaded pretrained! (epoch {} Acc@1 {})"
        #     .format(checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        raise RuntimeError(f'Not exist ==> {pth}')


def filter_best_model(best_dir: Path, prefix_name: str):
    best_path = sorted(best_dir.glob(prefix_name), key=lambda x: re.findall(r'Acc(.*?)_', str(x))[0], reverse=True)[0]
    return best_path.absolute()


def get_meters(names):
    return {k: AverageMeter() for k in names}


def log(msg, color=None, p_bar=False, current_batch=0, total_batch=0):
    if isinstance(color, str):
        color = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "cyan": 36,
            "white": 37
        }[color]
    if color is not None:
        msg = "\033[%dm%s\033[0m" % (color, msg)
    if p_bar:
        progress_bar(current_batch, total_batch, msg)
    else:
        print(msg)


class Chain(object):
    """it can be called by chain"""

    def __init__(self, _dict=None):
        if _dict:
            self.var = _dict
        else:
            self.var = {}

    def merge(self, _dict):
        self.var = {**self.var, **_dict}

    def __getattr__(self, item):
        return self.var[item]


def get_mean_and_std(dataset, batch_size=128, num_workers=2):
    '''
    Compute the mean and std value of dataset.
    transform = transforms.Compose([transforms.ToTensor()]) must be used in dataset.
    data: shape = (N,C,W,H)
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    bt_mean = []
    bt_std = []
    for inputs, _ in tqdm(loader, desc='compute mean and std'):
        inputs = inputs.to(device)
        bt_mean.append(inputs.mean(dim=(-2, -1)))
        bt_std.append(inputs.std(dim=(-2, -1)))

    mean = torch.cat(bt_mean).mean(dim=0)
    std = torch.cat(bt_std).mean(dim=0)

    return mean, std


def progress_bar(current, total, msg=None):
    TOTAL_BAR_LENGTH = 20.
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    sys.stdout.write(msg)
    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
