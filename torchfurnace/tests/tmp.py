# -*- coding: utf-8 -*-
# Date: 2020/3/17 16:24

"""
module description
"""
import shutil

__author__ = 'tianyu'

from torchfurnace.utils.function import get_meters, Chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

tb = SummaryWriter()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# tb.add_graph(Net(), torch.rand(1,1, 28, 28))
# for i in range(50):
#     x = np.random.random(1000)
#     tb.add_histogram('distribution centers', x, i)
#
#     x=np.random.rand(1000)
#     tb.add_histogram('distribution gussian', x, i)
#
#     x=np.random.gamma(1000)
#     tb.add_histogram('distribution gamma', x, i)
#
#
# tb.close()
a = Net()


# print(str(getattr(a,'__class__')).split(".")[-1].split("'")[0])
# import time

# print(time.strftime('%m%d_%H-%M-%S', time.localtime(time.time())))

#
# print(dir(a))
# print(list(filter(lambda attr: isinstance(attr, nn.Module), [getattr(a, attr) for attr in dir(a)])))
# hh=str(a)
# print(hh)
# print(Net().__class__.__name__)
# print(list(map(lambda x:x[4:],filter(lambda att:str(att).startswith('add_'),dir(tb)))))
# msg_key = list(map(lambda x: x[4:], filter(lambda att: str(att).startswith('add_'), dir(SummaryWriter))))
# print(msg_key)


class A():
    override = ['gg']

    def __setattr__(self, key, value):
        if key in self.override:
            self.__dict__[key] = value
        else:
            raise AttributeError("Don't override")

    def hh(self):
        print('hh')

    def gg(self):
        print('gg')


class B(A):
    def xx(self):
        print('xx')

    def hh(self):
        print('son hh')

    def gg(self):
        print('sone gg')


b = B()
a=A()
a.cc='sss'
b.hh(),b.xx(),b.gg()