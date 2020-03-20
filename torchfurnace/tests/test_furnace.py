# -*- coding: utf-8 -*-
# Date: 2020/3/17 15:33

"""
module description
"""
__author__ = 'tianyu'
import torch

import sys
from pathlib import Path

print('add path:', str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(f'Testing {Path(__file__)}')

from torchfurnace import Engine, Parser
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

parser = Parser('AKD-EEAP')
parser.add_argument('--add_test', default='test', type=str)
args = parser.parse_args()
experiment_name = '_'.join([args.dataset, ])

# eng = Engine(parser).experiment_name(experiment_name)


# model = models.alexnet(pretrained=False)
#
# train_ds = datasets.MNIST(
#     "data/mnist",
#     train=True,
#     download=True,
#     transform=transforms.Compose(
#         [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#     ),
# )
#
# val_ds = datasets.MNIST(
#     "data/mnist",
#     train=False,
#     download=True,
#     transform=transforms.Compose(
#         [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#     ),
# )
#
# optimizer = torch.optim.Adam(model.parameters())
# eng.learning(model, optimizer, train_ds, val_ds)
