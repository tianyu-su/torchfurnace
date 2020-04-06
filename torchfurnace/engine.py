# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:16

"""
an engine for deep learning task
"""
__author__ = 'tianyu'
import abc
import random
import time
import warnings

import numpy as np
import torch.backends.cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from .options import Parser
from .tracer import Tracer
from .utils import tracer_component as tc
from .utils.function import *


class Engine(object, metaclass=abc.ABCMeta):
    """
    Suggest Overriding Function:
    _on_start_epoch:           add some your meters for learning
    _get_lr_scheduler:         define your lr scheduler, default StepLR(step=30,gamma=0.1)
    _on_start_batch:           define how to read your dataset to return input,target as well as put on right device
    _add_on_end_batch_log:     add some your log information
    _add_on_end_batch_tb:      add some your visualization for tensorboard by add_xxx
    _add_record:               add some record information
    _before_evaluate:          define your operation before calling _validate evaluation mode
    _after_evaluate:           define your operation after calling _validate evaluation mode
    """

    def __init__(self, parser: Parser, experiment_name='exp'):
        self._parser = parser
        self._switch_training = True
        self._meters = self._status_meter()
        self._state = {'best_acc1': -1, 'training_iterations': 0, 'iteration': 0}
        self._experiment_name = experiment_name
        self._init_learning()

    def _status_meter(self):
        outer = self

        class StatusMeter(object):
            def __init__(self):
                self._training = Chain()
                self._validation = Chain()

            def __getattr__(self, item):
                if outer._switch_training:
                    return getattr(self._training, item)
                else:
                    return getattr(self._validation, item)

        return StatusMeter()

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

        if self._args.gpu is not None:
            # torch.backends.cudnn.benchmark = True
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self._args.gpu)
            # assign 0 because if you code os.environ['CUDA_VISIBLE_DEVICES']=xx,
            # all gpu device is 0 in pytorch context, otherwise you will get a
            # RuntimeError: CUDA error: invalid device ordinal
            self._args.gpu = 0

        if self._args.evaluate:
            self._args.p_bar = True
            self._args.no_tb = False

        if self._args.p_bar:
            self._args.print_freq = 1

    def _warp_loader(self, training, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self._args.batch_size, num_workers=self._args.workers,
                                           pin_memory=True, shuffle=training)

    def _init_learning(self):
        self._args = self._parser.parse_args()
        self._do_args()

        self._tracer = \
            Tracer(root_dir=Path(self._args.work_dir), work_name=self._parser.work_name, clean_up=self._args.clean_up) \
                .tb_switch(self._args.no_tb) \
                .debug_switch(self._args.debug or self._args.p_bar) \
                .attach(experiment_name=self._experiment_name, override=self._args.nowtime_exp,
                        logger_name=self._args.logger_name)

    @property
    def tracer(self):
        return self._tracer

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
    def _get_lr_scheduler(optimizer: torch.optim.Optimizer) -> list:
        return [StepLR(optim, 30, gamma=0.1) for optim in ([optimizer] if not isinstance(optimizer, list) else optimizer)]

    @staticmethod
    def _on_start_epoch():
        """
        add your meters by get_meters function
        for example : get_meters(['mine1', 'mine2'])
        usage:  self._meters[mode].{name}.update()  detail in : from .meter import AverageMeter
        """

        return get_meters([])

    def _add_record(self, ret_forward, batch_size):
        """
        self._meters.losses.update(ret['loss'], bs)
        """
        pass

    def _before_evaluate(self, model):
        """
        load checkpoint
        """
        for pth, m in zip(self._args.evaluate, [model] if not isinstance(model, list) else model):
            if os.path.isfile(pth):
                log("=> loading checkpoint '{}'".format(pth))
                checkpoint = torch.load(pth, map_location='cpu')
                m.load_state_dict(checkpoint['state_dict'])
                log("=> loaded checkpoint '{}' (epoch {} Acc@1 {})"
                    .format(pth, checkpoint['epoch'], checkpoint['best_acc1']))
            else:
                assert False, "=> no checkpoint found at '{}'".format(pth)

    def _after_evaluate(self):
        """
        execute something after evaluation
        """
        pass

    def _on_end_epoch(self, model, optimizer, is_best):
        """save more than one model and optimizer, for example GAN"""
        postfix = f'_{self._args.extension}'
        if self._args.extension == '': postfix = ''
        for m, optim in zip([model] if not isinstance(model, list) else model,
                            [optimizer] if not isinstance(optimizer, list) else optimizer):
            self._tracer.store(tc.Model(
                f"{model.__class__.__name__}{postfix}.pth.tar",
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
        if self._args.gpu is not None:
            return inp.cuda(self._args.gpu), target.cuda(self._args.gpu), target.size(0)
        else:
            return inp, target, target.size(0)

    def _add_on_end_batch_log(self, training):
        """ user can add some log information with _on_start_epoch using all kinds of meters in _on_end_batch"""
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

    def _on_end_batch(self, data_loader, optimizer=None):
        """ print log and visualization"""
        training_iterations = self._state['training_iterations']
        if self._switch_training:
            if self._state['iteration'] != 0 and self._state['iteration'] % self._args.print_freq == 0:
                print_process_bar = {'p_bar': self._args.p_bar, 'current_batch': self._state['iteration'], 'total_batch': len(data_loader)}
                if self._args.p_bar:
                    prefix_info = "Epoch:[{0}] "
                else:
                    prefix_info = 'Epoch: [{0}][{1}/{2}]\t'

                fix_log = prefix_info + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                fix_log = fix_log.format(
                    self._state['epoch'], self._state['iteration'], len(data_loader), batch_time=self._meters.batch_time,
                    data_time=self._meters.data_time, loss=self._meters.losses,
                    top1=self._meters.top1, top5=self._meters.top5)

                log(fix_log + self._add_on_end_batch_log(True), **print_process_bar)
                if self._args.no_tb:
                    self._tracer.tb.add_scalars('data/loss', {
                        'training': self._meters.losses.avg,
                    }, training_iterations)
                    self._tracer.tb.add_scalar('data/epochs', self._state['epoch'], training_iterations)
                    for oi, optim in enumerate([optimizer] if not isinstance(optimizer, list) else optimizer):
                        self._tracer.tb.add_scalars(f'data/learning_rate', {f'lr_optim_{oi + 1}': optim.param_groups[-1]['lr']}, training_iterations)
                    self._tracer.tb.add_scalars('data/precision/top1', {
                        'training': self._meters.top1.avg,
                    }, training_iterations)
                    self._tracer.tb.add_scalars('data/precision/top5', {
                        'training': self._meters.top5.avg
                    }, training_iterations)
                    self._tracer.tb.add_scalars('data/runtime', {
                        'batch_time': self._meters.batch_time.avg,
                        'data_time': self._meters.data_time.avg
                    }, training_iterations)
                    self._add_on_end_batch_tb(True)
        elif not self._args.evaluate:
            fix_log = ('Testing: Epoch [{0}]  Acc@1 {top1.avg:.3f}\tAcc@5 {top5.avg:.3f}\tLoss {loss.avg:.4f}\t[best:{best_acc}]\t'
                       .format(self._state['epoch'], top1=self._meters.top1, top5=self._meters.top5,
                               loss=self._meters.losses, best_acc=self._state['best_acc1']))
            log(fix_log + self._add_on_end_batch_log(False), green=True)
            if self._args.no_tb:
                self._tracer.tb.add_scalars('data/loss', {
                    'validation': self._meters.losses.avg,
                }, training_iterations)
                self._tracer.tb.add_scalars('data/precision/top1', {
                    'validation': self._meters.top1.avg,
                }, training_iterations)
                self._tracer.tb.add_scalars('data/precision/top5', {
                    'validation': self._meters.top5.avg
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
        ret = {'loss': object, 'acc1': object, 'acc5': object}

        # do something
        output = model(inp)
        loss = F.cross_entropy(output, target)

        # compute acc1 acc5
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ret['loss'] = loss.item()
        ret['acc1'] = acc1.item()
        ret['acc5'] = acc5.item()

        return ret

    def _train(self, model, train_loader, optimizer, epoch):
        self._switch_training = True

        # setup model
        [m.train() for m in (model if isinstance(model, list) else [model])]

        self._meters.merge(get_meters(['batch_time', 'data_time', 'losses', 'top1', 'top5']))
        self._meters.merge(self._on_start_epoch())

        end = time.time()

        for i, batch in enumerate(train_loader):
            self._state['training_iterations'] += 1
            self._state['iteration'] = i
            self._state['epoch'] = epoch
            # measure data loading time
            self._meters.data_time.update(time.time() - end)

            inp, target, bs = self._on_start_batch(batch)

            # compute output
            ret = self._on_forward(True, model, inp, target, optimizer)

            # record indicators
            self._meters.losses.update(ret['loss'], bs)
            self._meters.top1.update(ret['acc1'], bs)
            self._meters.top5.update(ret['acc5'], bs)
            self._add_record(ret, bs)

            # measure elapsed time
            self._meters.batch_time.update(time.time() - end)
            end = time.time()

            self._on_end_batch(train_loader, optimizer)

    def _validate(self, model, val_loader):
        self._switch_training = False

        # setup model
        [m.eval() for m in (model if isinstance(model, list) else [model])]

        self._meters.merge(get_meters(['batch_time', 'losses', 'top1', 'top5']))
        self._meters.merge(self._on_start_epoch())

        end = time.time()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                self._state['iteration'] = i

                inp, target, bs = self._on_start_batch(batch)

                # compute output
                ret = self._on_forward(False, model, inp, target)

                # record indicators
                self._meters.losses.update(ret['loss'], bs)
                self._meters.top1.update(ret['acc1'], bs)
                self._meters.top5.update(ret['acc5'], bs)
                self._add_record(ret, bs)

                # measure elapsed time
                self._meters.batch_time.update(time.time() - end)
                end = time.time()

            self._on_end_batch(val_loader)
        return self._meters.top1.avg

    def learning(self, model, optimizer, train_dataset, val_dataset):
        """
        Core function of engine to organize training process
        :param val_dataset: training dataset
        :param train_dataset: validation dataset
        :param model: one or list
        :param optimizer: one or list
        """

        # save config
        cfg = {f"optimizer{i + 1}": optim for i, optim in enumerate([optimizer] if not isinstance(optimizer, list) else optimizer)}
        self._tracer.store(tc.Config({**cfg, **vars(self._args)}))

        train_loader = self._warp_loader(True, train_dataset)
        val_loader = self._warp_loader(False, val_dataset)

        log('==> Start ...', green=True)
        if self._args.resume:
            self._resume(model, optimizer)

        # cuda setup
        if self._args.gpu is not None:
            [m.cuda(self._args.gpu) for m in (model if isinstance(model, list) else [model])]

        if self._args.evaluate:
            self._before_evaluate(model)
            self._validate(model, val_loader)
            self._after_evaluate()
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

            print(f"Best Acc1:{self._state['best_acc1']}")
            self._close()
            return self._state['best_acc1']
