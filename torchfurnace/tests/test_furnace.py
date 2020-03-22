# -*- coding: utf-8 -*-
# Date: 2020/3/17 15:33

"""
module description
"""
__author__ = 'tianyu'

import sys
from pathlib import Path

print('add path:', str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(f'Testing {Path(__file__)}')

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torchfurnace import Engine, Parser
from torchfurnace.utils.decorator import test_function

# define experiment
parser = Parser('TVGG11')
args = parser.parse_args()
args.dataset = 'CIFAR10'
experiment_name = '_'.join([args.dataset, args.exp_suffix])


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


# trick
import platform
import hashlib

m = hashlib.md5()
m.update((platform.platform() + platform.node()).encode("utf8"))
if m.hexdigest() == 'f8f97b731ed522f9bbfbc89a4c828e05':
    datasets.CIFAR10.url = 'file:///D:/Downloads/cifar-10-python.tar.gz'


class TmpDataset4Test(datasets.CIFAR10):
    def __len__(self):
        return 2


train_ds = TmpDataset4Test("dataset/cifar10", train=True, download=True,
                           transform=transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]))
val_ds = TmpDataset4Test("dataset/cifar10", train=False, download=True,
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]))

model = models.vgg11(pretrained=False, num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


@test_function
def test_learning():
    global parser, experiment_name, args
    eng = VGGNetEngine(parser).experiment_name(experiment_name)
    sys.argv.extend(['--epochs', '1'])
    acc1 = eng.learning(model, optimizer, train_ds, val_ds)
    print('Acc1:', acc1)


@test_function
def test_validation():
    global parser, experiment_name
    eng = VGGNetEngine(parser).experiment_name(experiment_name)
    sys.argv.extend(['--epochs', '1'])
    sys.argv.extend(['-eval'])
    sys.argv.extend(['--resume', 'CIFAR10/VGG_Epk1_Acc0.00_best.pth.tar'])
    eng.learning(model, optimizer, train_ds, val_ds)


@test_function
def toy_save():
    global model
    testloader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)

            test_loss += loss.item()

    print(f'Saving..    Loss:{test_loss}')
    state = {
        'net': model.state_dict()
    }
    torch.save(state, 'ckpt.pth')


@test_function
def toy_load():
    global model
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('ckpt.pth')
    model.load_state_dict(checkpoint['net'])

    testloader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)

            test_loss += loss.item()

    print(f'Loaded..    Loss:{test_loss}')


if __name__ == '__main__':
    sys.argv.append('--debug')
    sys.argv.append('--nowtime_exp')

    test_learning()
    test_validation()

    # The loss are different between save_model and load_model.
    # Why ????
    toy_save()
    toy_load()
