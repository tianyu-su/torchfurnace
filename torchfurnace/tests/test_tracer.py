# -*- coding: utf-8 -*-
# Date: 2020/3/18 23:12

"""
module description
"""
import torch

__author__ = 'tianyu'

import sys
from pathlib import Path

print('add path:', str(Path(__file__).absolute().parent.parent.parent))

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(f'Testing {Path(__file__)}')

from torchfurnace import Tracer
import torchfurnace.utils.tracer_component as tc
from pathlib import Path
import torchvision.models as models
from torchfurnace import Parser
from torchfurnace.utils.decorator import test_function

tracer = Tracer(Path(r'.'), 'mine_network').tb_switch(True).attach('expn')
model = models.vgg11(pretrained=False)
optimizer = torch.optim.Adam(model.parameters())


@test_function
def test_config():
    global tracer, model, optimizer
    optimizer1 = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.99)
    cfg = {f"optimizer{i + 1}": optim for i, optim in enumerate([optimizer, optimizer1])}
    tracer.store(tc.Config({**cfg, **vars(Parser('mine_network').parse_args())}))


@test_function
def test_store_model():
    global tracer, model, optimizer
    tracer.store(tc.Model(
        f"{model.__class__.__name__}_Epk{99}_Acc{0.66}_extinfo.pth.tar",
        {
            'epoch': 99,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc1': 0.66,
            'optimizer': optimizer.state_dict(),
        }, True))


@test_function
def test_load_model():
    global tracer, model, optimizer
    pth = f"expn/{model.__class__.__name__}_Epk{99}_Acc{0.66}_extinfo_best.pth.tar"
    ret = tracer.load(tc.Model(
        pth, {
            'model': model,
            'optim': optimizer
        }))
    print(ret)


@test_function
def test_tensorboard():
    global tracer, model, optimizer
    import numpy as np

    [tracer.tb.add_scalar('data/epochs', y, x) for y, x in zip(np.random.randint(50, 90, 100), np.linspace(0, 99, 100, dtype=np.int))]


if __name__ == '__main__':
    test_config()

    test_store_model()
    test_load_model()

    test_tensorboard()

    tracer.close()
