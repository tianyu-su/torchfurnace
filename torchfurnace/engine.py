# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:16

"""
an engine for deeplearning task
"""
__author__ = 'tianyu'
import abc
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch.backends.cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .options import Parser
from .tracer import Tracer
from .utils import tracer_component as tc
from .utils.decorator import *
from .utils.function import *


class Engine(object, metaclass=abc.ABCMeta):
    """
    Suggest Overriding Function:
    _on_start_epoch:           add some your meters for learning
    _get_lr_scheduler:         define your lr scheduler, default StepLR(step=30,gamma=0.1)
    _on_start_batch:           define how to read your dataset to return input,target as well as put on right device
    _add_on_end_batch_log:     add some your log information
    _add_on_end_batch_tb:      add some your visualization for tensorboard by add_xxx
    """

    def __init__(self, parser: Parser):
        self._parser = parser
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._meters = {'training': Chain(), 'validation': Chain()}
        self._state = {'best_acc1': -1, 'training_iterations': 0, 'iteration': 0}
        self._experiment_name = 'exp'

    def experiment_name(self, name):
        self._experiment_name = name
        return self

    def _close(self):
        self._tracer.close()

    def _do_args(self):
        if self._args.deterministic:
            seed = 1541233595
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.set_printoptions(precision=10)
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
        if self._args.debug:
            self._args.workers = 0
            self._args.batch_size = 2

        if torch.cuda.is_available():
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = self._args.gpu
            torch.backends.cudnn.benchmark = True

    def _warp_loader(self, training, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self._args.batch_size, num_workers=self._args.workers,
                                           pin_memory=True, shuffle=training)

    def _init_learning(self):
        self._args = self._parser.parse_args()
        self._do_args()

        self._tracer = Tracer(root_dir=Path(self._args.work_dir), work_name=self._parser.work_name) \
            .tb_switch(self._args.no_tb) \
            .debug_switch(self._args.debug or self._args.p_bar) \
            .attach(experiment_name=self._experiment_name, override=self._args.override_exp,
                    logger_name=self._args.logger_name)

    def _resume(self, model, optimizer):
        """load more than one model and optimizer, for example GAN"""
        for pth, m, optim in zip(self._args.resume, [model] if not isinstance(model, list) else model,
                                 [optimizer] if not isinstance(optimizer, list) else optimizer):
            ret = self._tracer.load(tc.Model(
                pth, {
                    'model': m,
                    'optim': optim
                }))
            self._args.start_epoch = ret['start_epoch']
            self._state['best_acc1'] = ret['best_acc1']
        self._args.epochs += self._args.start_epoch

    @staticmethod
    def _get_lr_scheduler(optimizer: object) -> list:
        return [StepLR(optim, 30, gamma=0.1) for optim in ([optimizer] if not isinstance(optimizer, list) else optimizer)]

    @staticmethod
    def _on_start_epoch():
        """add your meters by get_meters function """
        return get_meters([])

    def _on_end_epoch(self, model, optimizer, is_best):
        """save more than one model and optimizer, for example GAN"""
        postfix = f'_{self._args.extension}'
        if self._args.extension == '': postfix = ''
        for m, optim in zip([model] if not isinstance(model, list) else model,
                            [optimizer] if not isinstance(optimizer, list) else optimizer):
            self._tracer.store(tc.Model(
                f"{model.__class__.__name__}_Epk{self._state['epoch'] + 1}_Acc{self._state['best_acc1']:.2f}{postfix}.pth.tar",
                {
                    'epoch': self._state['epoch'] + 1,
                    'arch': str(m),
                    'state_dict': m.state_dict(),
                    'best_acc1': self._state['best_acc1'],
                    'optimizer': optim.state_dict(),
                }, is_best))

    def _on_start_batch(self, data):
        """override to adapt yourself dataset __getitem__"""
        inp, target = data
        return inp.to(self._device), target.to(self._device)

    def _add_on_end_batch_log(self, training):
        """ user can add some log information with _on_start_epoch using all kinds of meters"""
        if training:
            pass
        else:
            pass
        return ""

    def _add_on_end_batch_tb(self, training):
        """ user can add some tensorboard operations with _on_start_epoch using all kinds of meters"""
        if training:
            pass
        else:
            pass

    def _on_end_batch(self, training, data_loader, optimizer=None):
        """ print log and visualization"""
        mode = 'training' if training else "validation"
        training_iterations = self._state['training_iterations']
        if training:
            if self._state['iteration'] != 0 and self._state['iteration'] != 0 % self._args.print_freq == 0:
                fix_log = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t ' \
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f}) '.format(
                    self._state['epoch'], self._state['iteration'], len(data_loader), batch_time=self._meters[mode].batch_time,
                    data_time=self._meters[mode].data_time, loss=self._meters[mode].losses,
                    top1=self._meters[mode].top1, top5=self._meters[mode].top5)
                log(fix_log + self._add_on_end_batch_log(True))
                if self._args.no_tb:
                    self._tracer.tb.add_scalars('data/loss', {
                        'training': self._meters[mode].losses.avg,
                    }, training_iterations)
                    self._tracer.tb.add_scalar('data/epochs', self._state['epoch'], training_iterations)
                    for oi, optim in enumerate([optimizer] if not isinstance(optimizer, list) else optimizer):
                        self._tracer.tb.add_scalars(f'data/learning_rate', {f'lr_optim_{oi + 1}': optim.param_groups[-1]['lr']}, training_iterations)
                    self._tracer.tb.add_scalars('data/precision/top1', {
                        'training': self._meters[mode].top1.avg,
                    }, training_iterations)
                    self._tracer.tb.add_scalars('data/precision/top5', {
                        'training': self._meters[mode].top5.avg
                    }, training_iterations)
                    self._tracer.tb.add_scalars('data/runtime', {
                        'batch_time': self._meters[mode].batch_time.avg,
                        'data_time': self._meters[mode].data_time.avg
                    }, training_iterations)
                    self._add_on_end_batch_tb(True)


        else:
            fix_log = ('Testing: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f} '
                       .format(top1=self._meters[mode].top1, top5=self._meters[mode].top5, loss=self._meters[mode].losses))
            log(fix_log + self._add_on_end_batch_log(False), green=True)
            if self._args.no_tb:
                self._tracer.tb.add_scalars('data/loss', {
                    'validation': self._meters[mode].losses.avg,
                }, training_iterations)
                self._tracer.tb.add_scalars('data/precision/top1', {
                    'validation': self._meters[mode].top1.avg,
                }, training_iterations)
                self._tracer.tb.add_scalars('data/precision/top5', {
                    'validation': self._meters[mode].top5.avg
                }, training_iterations)
                self._add_on_end_batch_tb(False)

    @staticmethod
    @abc.abstractmethod
    def _on_forward(training, model, inp, target, optimizer=None) -> dict:
        """
        implement training and validation code here
        :param training: bool -> training validation
        :param model: one or list
        :param inp: batch data
        :param target: batch target
        :param optimizer: one or list
        :return:
        """

        """ for example """
        # ret can expand but DONT Shrink
        ret = {'loss': object, 'preds': object}

        # do something
        output = model(inp)
        loss = F.cross_entropy(output, target)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ret['loss'] = loss
        ret['preds'] = output

        return ret
        # raise NotImplementedError

    @train_wrapper
    def _train(self, model, train_loader, optimizer, epoch):
        mode = 'training'
        self._meters[mode].merge(get_meters(['batch_time', 'data_time', 'losses', 'top1', 'top5']))
        self._meters[mode].merge(self._on_start_epoch())

        if self._args.p_bar:
            train_loader = tqdm(train_loader, desc='Training')
        end = time.time()

        for i, batch in enumerate(train_loader):
            self._state['training_iterations'] += 1
            self._state['iteration'] = i
            self._state['epoch'] = epoch
            # measure data loading time
            self._meters[mode].data_time.update(time.time() - end)

            inp, target = self._on_start_batch(batch)

            # compute output
            ret = self._on_forward(True, model, inp, target, optimizer)

            # compute acc1 acc5
            acc1, acc5 = accuracy(ret['preds'], target, topk=(1, 5))
            self._meters[mode].losses.update(ret['loss'].item(), inp.size(0))
            self._meters[mode].top1.update(acc1[0], inp.size(0))
            self._meters[mode].top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            self._meters[mode].batch_time.update(time.time() - end)
            end = time.time()

            self._on_end_batch(True, train_loader, optimizer)

    @val_wrapper
    def _validate(self, model, val_loader):
        mode = 'validation'
        self._meters[mode].merge(get_meters(['batch_time', 'losses', 'top1', 'top5']))
        self._meters[mode].merge(self._on_start_epoch())

        if self._args.p_bar:
            val_loader = tqdm(val_loader, desc='Validation')
        end = time.time()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                self._state['iteration'] = i

                inp, target = self._on_start_batch(batch)

                # compute output
                ret = self._on_forward(False, model, inp, target)
                # compute acc1 acc5
                acc1, acc5 = accuracy(ret['preds'], target, topk=(1, 5))
                self._meters[mode].losses.update(ret['loss'].item(), inp.size(0))
                self._meters[mode].top1.update(acc1[0], inp.size(0))
                self._meters[mode].top5.update(acc5[0], inp.size(0))

                # measure elapsed time
                self._meters[mode].batch_time.update(time.time() - end)
                end = time.time()

            self._on_end_batch(False, val_loader)
        return self._meters[mode].top1.avg

    def learning(self, model, optimizer, train_dataset, val_dataset):
        """
        Core function of engine to organize training process
        :param val_dataset: training dataset
        :param train_dataset: validation dataset
        :param model: one or list
        :param optimizer: one or list
        """
        self._init_learning()

        # save config
        cfg = {f"optimizer{i + 1}": optim for i, optim in enumerate([optimizer] if not isinstance(optimizer, list) else optimizer)}
        self._tracer.store(tc.Config({**cfg, **vars(self._args)}))

        train_loader = self._warp_loader(True, train_dataset)
        val_loader = self._warp_loader(False, val_dataset)

        log('==> Start ...', green=True)
        if self._args.resume:
            self._resume(model, optimizer)

        if self._args.evaluate:
            self._validate(model, val_loader)
        else:
            ajlr = None
            if self._args.adjust_lr:
                ajlr = self._get_lr_scheduler(optimizer)
            for epoch in range(self._args.start_epoch, self._args.epochs):

                # train for one epoch
                self._train(model, train_loader, optimizer, epoch)

                # evaluate on validation set
                acc1 = self._validate(model, val_loader)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > self._state['best_acc1']
                self._state['best_acc1'] = max(acc1, self._state['best_acc1'])

                self._on_end_epoch(model, optimizer, is_best)

                if self._args.adjust_lr:
                    [lr.step() for lr in ajlr]
        self._close()
        return self._state['best_acc1']
