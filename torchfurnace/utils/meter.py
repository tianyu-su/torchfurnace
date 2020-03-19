# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:02

"""
some meter for measuring model performance
"""
__author__ = 'tianyu'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"sum={self.sum} count={self.count} average={self.avg} last_value={self.val}"
