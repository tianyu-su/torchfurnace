# -*- coding: utf-8 -*-
# Date: 2020/3/17 17:05

"""
module description
"""
__author__ = 'tianyu'
import functools

import torch
import torch.nn as nn


def train_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            for arg in args:
                if isinstance(arg, list):
                    [var.cuda() or var.train() for var in arg if isinstance(var, nn.Module)]
                elif isinstance(arg, nn.Module):
                    arg.cuda(), arg.train()
        return func(*args, **kwargs)

    return wrapper


def val_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            for arg in args:
                if isinstance(arg, list):
                    [var.cuda() or var.eval() for var in arg if isinstance(var, nn.Module)]
                elif isinstance(arg, nn.Module):
                    arg.cuda(), arg.eval()
        return func(*args, **kwargs)

    return wrapper
