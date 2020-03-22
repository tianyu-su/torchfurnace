# torchfurnace [![Build Status](https://travis-ci.com/tianyu-su/torchfurnace.svg?branch=master)](https://travis-ci.com/tianyu-su/torchfurnace) ![](https://img.shields.io/badge/pytorch-1.1.0-blue) ![](https://img.shields.io/badge/python-3.6-blue)

`torchfurnace` is a tool package for training model, pre-processing dataset and managing experiment record in pytorch AI tasks.

## Quick Start

### Usage

`pip install torchfurnace`

### Example
trainig VGG16 for CIFAR10

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import MultiStepLR
from torchfurnace import Engine, Parser

# define training process of your model
class VGGNetEngine(Engine):
    @staticmethod
    def _on_forward(training, model, inp, target, optimizer=None) -> dict:
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

    @staticmethod
    def _get_lr_scheduler(optim) -> list:
        return [MultiStepLR(optim, milestones=[150, 250, 350], gamma=0.1)]

def main():
    # define experiment name
    parser = Parser('TVGG16')
    args = parser.parse_args()
    experiment_name = '_'.join([args.dataset, args.exp_suffix])

    # Data
    ts = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = CIFAR10(root='data', train=True, download=True, transform=ts)
    testset = CIFAR10(root='data', train=False, download=True, transform=ts)

    # define model and optimizer
    net = torchvision.models.vgg16(pretrained=False, num_classes=10)
    net.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
    net.classifier = nn.Linear(512, 10)
    optimizer = torch.optim.Adam(net.parameters())

    # new engine instance
    eng = VGGNetEngine(parser).experiment_name(experiment_name)
    acc1 = eng.learning(net, optimizer, trainset, testset)
    print('Acc1:', acc1)

if __name__ == '__main__':
    import sys
    run_params='--dataset CIFAR10 -lr 0.1 -bs 128 -j 2 --epochs 400 --adjust_lr'
    sys.argv.extend(run_params.split())
    main()
```

## Introduction
### Why do this?
There are some deep learning frameworks to quickly build a training system in pytorch AI tasks, however, I found that most of them are complex framework which have higher cost for learning it and seriously invade original code , for instance, maybe modify your model class to adapt the framework.

So, `torchfurnace` is born for perform your pytorch AI task quickly, simply and without invasion viz you don't have to change too much defined code.

### What features?
1. `torchfurnace` consists of two independent components including `engine` and `tracer`. `engine` is a core component of proposed framework, and `tracer` is a manager of experiment whose obligation include log writing, model saving and training visualization.

2. `torchfurnace` integrates some practical tools, such as processing raw dataset to LMDB for solving I/O bottleneck and computing the number of parameter size.

### Components

#### Engine

```python
from torchfurnace import Engine
```

#### Tracer
```python
from torchfurnace import Tracer
```

#### Parser
```python
from torchfurnace import Parser
```

#### ImageFolderLMDB
```python
from torchfurnace import ImageFolderLMDB
```

#### ImageLMDB
```python
from torchfurnace import ImageLMDB
```

#### Model Summary
This tool comes from [pytorch-summary](https://github.com/sksq96/pytorch-summary/).

```python
import torchvision
from torchfurnace.utils.torch_summary import summary, summary_string
net = torchvision.models.vgg16()

# this function will output result on screen.  
summary(net,(3,224,224))

# this funcion will return a string of description.
summary_string(net,(3,224,224))

```

## Directory Architecture
```text
TVGG16/
├── logs
│   └── CIFAR10
│       └── log.txt
├── models
│   └── CIFAR10
│       ├── architecture.txt
│       ├── checkpoint
│       │   └── best
│       └── run_config.json
└── tensorboard
    └── CIFAR10
        └── events.out.tfevents
```


## Testing & Example
In this section, you have to `git clone https://github.com/tianyu-su/torchfurnace.git`. 

1. `torchfurnace/tests/test_furnace.py` A unit test for `Engine`.
2. `torchfurnace/tests/test_tracer.py` A unit test for `Tracer`.
3. `torchfurnace/tests/test_img2lmdb.py` A unit test for convert images to LMDB.
4. `torchfurnace/tests/test_vgg16.py` A compare experiment with [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py) to validate pipeline of the proposed framework. 

## More Usages
1. `options.py`，flags: no_tb, p_bar, override ,ext ,exp_suffix


# TODO

- [ ] training by `DistributedDataParallel`
- [ ] compute mean and standard deviation of image dataset

