# -*- coding: utf-8 -*-
# Date: 2020/3/20 23:20

"""
test framework efficient
setup: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
"""
import torchvision

__author__ = 'tianyu'
import sys
from pathlib import Path

print('add path:', str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(f'Testing {Path(__file__)}')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchfurnace import Engine, Parser
from torchfurnace.utils.decorator import test_function

# define experiment
parser = Parser('TVGG16')
args = parser.parse_args()
args.dataset = 'CIFAR10'
experiment_name = '_'.join([args.dataset, ])


class VGGNetEngine(Engine):
    @staticmethod
    def _on_forward(training, model, inp, target, optimizer=None) -> dict:
        # ret can expand but DONT Shrink
        ret = {'loss': object, 'preds': object}

        output = model(inp)
        loss = F.cross_entropy(output, target)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ret['loss'] = loss
        ret['preds'] = output

        return ret


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=False, download=True, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = VGG('VGG16')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

@test_function
def test_precision():
    global experiment_name
    eng = VGGNetEngine(parser).experiment_name(experiment_name)
    eng._args.batch_size = 128
    eng._args.workers = 2
    eng._args.epochs = 200
    acc1 = eng.learning(model, optimizer, trainset, testset)
    print('Acc1:', acc1)


@test_function
def test_img2lmdb_precision():
    pass
