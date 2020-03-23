# torchfurnace [![Build Status](https://travis-ci.com/tianyu-su/torchfurnace.svg?branch=master)](https://travis-ci.com/tianyu-su/torchfurnace) ![](https://img.shields.io/badge/pytorch-1.1.0-blue) ![](https://img.shields.io/badge/python-3.6-blue)

`torchfurnace` is a tool package for training model, pre-processing dataset and managing experiment record in pytorch AI tasks.

## Quick Start

### Usage

`pip install torchfurnace`

### Example
VGG16 for CIFAR10

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

2. `torchfurnace` integrates some practical tools, such as processing raw dataset to LMDB for solving I/O bottleneck and computing the number of parameter size and dataset with std and mean.

### Components

#### Engine

```python
from torchfurnace import Engine
```
This is core component of the proposed framework. `Engine` is an abstract class which controls the whole workflow to finish AI tasks. So, you have to implement `_on_forward` by inheriting the `Engine` when you start. The job of `_on_forward` is to define which criterion to use and how to optimize your model.

For example (a minimal configuration):

```python
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
```

Above,  you have done the preparatory work of framework, and then you define model, train dataset, val dataset and optimizer as usual. This highlights the advantages of our method which don't  heavily invade original code. In other words, you don't have to modify your code too much.  As follow, only 3 lines to finish learning. Of course, you can also assign a list type as model and optimizer, however, it's vital each element of models and optimizer must be matched.

```python
parser = Parser('TVGG11')
eng = VGGNetEngine(parser).experiment_name(experiment_name)
eng.learning(model, optimizer, train_ds, val_ds)
```



If you want more customizable features, you can override them with the following listed functions.

#####  overridden function

1. `_get_lr_scheduler`: This function support to define other scheduler in `torch.optim.lr_scheduler`. Default:  StepLR with gamma=0.1 and step_size=30
2. `_on_start_epoch`: You can add some meters , such as AverageMeter using `from torchfurnace.utils.meter import AverageMeter` in this function when you want to record more information in the training processing. 
3. `_on_start_batch`: define how to get input and target  from your dataset and put them on GPU.
4. `_add_on_end_batch_log`: You can add some log output information just like other meters in `_on_end_batch`. 
5. `_add_on_end_batch_tb`:  You can add some tensorboard output information just like other meters in `_on_end_batch`. 

#### Tracer

```python
from torchfurnace import Tracer
```

The  responsibility of this component is to manage  log writing, model saving and training visualization in experiment.

Especially, this component can run independently, meanwhile, it is integrated in `Engine` as a nested object to manage experiment record.

For example:

```python
from torchfurnace import Tracer
tracer = Tracer('my_network').attach('expn')
```

##### save experiment configuration

```python
import torchfurnace.utils.tracer_component as tc
cfg = {"optimizer": optim, 'addtion': 'hhh'}
tracer.store(tc.Config({**cfg, **vars(Parser('my_network').parse_args())}))
```

cfg: some customize informations 

Parser('my_network').parse_args() : parser snapshot

##### save checkpoint

```python
import torchfurnace.utils.tracer_component as tc
model = models.vgg11(pretrained=False)
optimizer = torch.optim.Adam(model.parameters())
tracer.store(tc.Model(
        f"{model.__class__.__name__}_extinfo.pth.tar",
        {
            'epoch': 99,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc1': 0.66,
            'optimizer': optimizer.state_dict(),
        }, True))
```

save checkpoint to appropriate location automatically.

##### load checkpoint

```python
import torchfurnace.utils.tracer_component as tc
model = models.vgg11(pretrained=False)
optimizer = torch.optim.Adam(model.parameters())
# 'pth' has fix format. It contains two parts, which are experiment file name and best model file name, skipping medial path , and they are separate by '/' .
pth = f"expn/{model.__class__.__name__}_Epk{99}_Acc{0.66}_extinfo_best.pth.tar"
ret = tracer.load(tc.Model(
        pth, {
            'model': model,
            'optim': optimizer
        }))
print(ret)
# ret has start_epoch and best_acc1
```

load checkpoint from appropriate location automatically.

##### display TensorBoard

```python
tracer.tb_switch(True)
tracer.tb_switch(False)
```

##### redirect output

```python
tracer.debug_switch(True)
tracer.debug_switch(False)
```

#### Parser

```python
from torchfurnace import Parser
```

This just is an  `ArgumentParser`, but have defined some frequently-used options.

##### print default options

```python
from torchfurnace import Parser
p = Parser()
args = p.parse_args()
print(args)
```

```json
{
    "dataset":"",
    "batch_size":1,
    "workers":2,
    "lr":0.01,
    "weight_decay":0.0005,
    "momentum":0.9,
    "dropout":0.5,
    "start_epoch":0,
    "epochs":200,
    "gpu":"0",
    "exp_suffix":"",
    "extension":"",
    "resume":"",
    "evaluate":false,
    "deterministic":false,
    "adjust_lr":false,
    "print_freq":10,
    "logger_name":"log.txt",
    "work_dir":"",
    "clean_up":5,
    "debug":false,
    "p_bar":false,
    "no_tb":true,
    "nowtime_exp":true
}
```

##### add option in Engine

```python
parser = Parser('TVGG16')
parser.add_argument('--add',default='addtion',type=str)
eng = Engine(parser).experiment_name(experiment_name)
```

#### ImageFolderLMDB

```python
from torchfurnace import ImageFolderLMDB
```

This component is a tool to accelerate data reading speed. Especially, your dataset folder should have a specific named regular just like the regular in `from torchvision.datasets import ImageFolder`

Like this:

```
demo_dataset/
├── train
│   ├── cat
│   │   └── 001.jpg
│   └── dog
│       └── 001.jpg
└── val
    ├── cat
    │   └── 001.jpg
    └── dog
        └── 001.jpg
```

##### make LMDB

```python
ImageFolderLMDB.folder2lmdb('demo_dataset', name='train', num_workers=16)
ImageFolderLMDB.folder2lmdb('demo_dataset', name='val', num_workers=16)
```

##### use LMDB

```python
from torch.utils.data import DataLoader
_set = ImageFolderLMDB('demo_dataset/train.lmdb')
loader = DataLoader(_set, batch_size=1, collate_fn=lambda x: x)
```

#### ImageLMDB

```python
from torchfurnace import ImageLMDB
```

This component is a tool for converting .jpg/.png to binary data and storing to LMDB. Because of reading lots of small size image always costing expensive I/O resources, it will improve speed of data loading.

##### make LMDB

```python
data_mapping = {
   'key': [f"{k}_{no:04}" for k in ['dog', 'cat'] for no in range(1, 4)],
   'pic_path': [f"demo_dataset/{k}{no:03}.jpg" for k in ['d', 'c'] for no in range(1, 4)]
}
ImageLMDB.store(data_mapping, 'demo_imgs_db', 4800000, num_workers=16)
```

`data_mapping` provide key and picture path.

##### read LMDB

```python
import matplotlib.pyplot as plt
read_keys = {
    'key': [f"{k}_{no:04}" for k in ['dog', 'cat'] for no in range(1, 4)]
}

for key in data_mapping['key']:
    img = ImageLMDB.read('demo_imgs_db', key)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```

The data from LMDB format is PIL object, so you can do other image processing, such as random crop, random flip and normalize using `torchvision.transforms.Compose`

#### Compute mean and std

Compute the mean and std value of dataset using GPU.

```python
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchfurnace.utils.function import get_mean_and_std

ts = transforms.Compose([transforms.ToTensor()])
trainset = CIFAR10(root='data', train=True, download=True, transform=ts)
mean, std = get_mean_and_std(trainset)
print(mean, std)
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
desc = summary_string(net,(3,224,224))
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

`TVGG16`: the root of your work name

`CIFAR10`: one of experiment name

`TVGG16/logs`: all kinds of experiments output files

`TVGG16/models`: all kinds of experiments checkpoints, network architectures and run configurations

`TVGG16/tensorboard`: all kinds of experiments outputs of tensorboard

## Testing & Example

In this section, you have to `git clone https://github.com/tianyu-su/torchfurnace.git`. 

1. `torchfurnace/tests/test_furnace.py` A unit test for `Engine`.
2. `torchfurnace/tests/test_tracer.py` A unit test for `Tracer`.
3. `torchfurnace/tests/test_img2lmdb.py` A unit test for convert images to LMDB.
4. `torchfurnace/tests/test_vgg16.py` A compare experiment with [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py) to validate pipeline of the proposed framework. 

## More Usages

Following methods are based on default setup.

1. open debug mode:  `--debug`  this mode is easy to debug.
2. close  tensorboard:  `--no_tb`
3. open process bar:  `--p_bar`
4. change total data directory: `--work_dir your_path`
5. remain best checkpoint Top k: `--clean_up k` 
6. run again with same setup, but default options will override old data. `--nowtime_exp`
7. make special marks for model checkpoint or other aspects: `--ext sp1`
8. make special marks for experiment name: `--exp_suffix`

Other  settings please look at [ops](#print-default-options).

# TODO

- [ ] `DistributedDataParallel` 

