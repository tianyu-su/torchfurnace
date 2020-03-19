# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:03

"""
some functions for learning model
"""

__author__ = 'tianyu'

import os

import torch

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


def load_pretrained(model, pth):
    if os.path.isfile(pth):
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['state_dict'])


def get_meters(names):
    return {k: AverageMeter() for k in names}


def log(msg, green=False):
    if green:
        print('\033[92m', end="")
    print(msg)
    print('\033[0m')


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

# def save_checkpoint(save_pth: Path(), state, best):
#     torch.save(state, save_pth)
#     if best:
#         shutil.copyfile(save_pth, os.path.join(save_pth.parent, 'best', save_pth.name.replace('.pth.tar', '_best.pth.tar')))
#

# def load_checkpoint(weights_pth, state):
#     ret = {'start_epoch': -1, 'best_acc1': -1}
#
#     if os.path.isfile(weights_pth):
#         log("=> loading checkpoint '{}'".format(weights_pth))
#         checkpoint = torch.load(weights_pth)
#         ret['start_epoch'] = checkpoint['epoch']
#         ret['best_acc1'] = checkpoint['best_acc1']
#         state['model'].load_state_dict(checkpoint['state_dict'])
#         state['optim'].load_state_dict(checkpoint['optimizer'])
#         log("=> loaded checkpoint '{}' (epoch {} Acc@1 {})"
#             .format(weights_pth, checkpoint['epoch'], checkpoint['best_acc1']))
#     else:
#         assert False, "=> no checkpoint found at '{}'".format(weights_pth)
#
#     return ret
