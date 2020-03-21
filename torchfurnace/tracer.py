# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:14

"""
a tool module for managing model checkpoints, tensorboard, and experiment records
The key idea is to deal with the structure of directory, we only consider what files are needed to save, and then
the file will be saved in the appropriate location.
"""

__author__ = 'tianyu'

import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .utils.tracer_component import Config, Model


class Tracer(object):
    CONFIG_NAME = 'run_config'
    ARCH_NAME = 'architecture'

    def __init__(self, root_dir=Path('.'), work_name='network'):
        self._tb_switch = True
        self._debug_switch = False
        self._dirs = {
            'work_name': root_dir / work_name
        }

    def _start_log(self, logger_name):
        # redirect stdout stderr to file
        self._log = open(logger_name, 'w', encoding='utf-8')
        self._stderr = sys.stderr
        self._stdout = sys.stdout
        sys.stderr = self._log
        sys.stdout = self._log

    def close(self):
        # close I/O
        if self._tb_switch: self._tb.close()
        if not self._debug_switch:
            self._log.close()
            # recovery stdout stderr to system
            sys.stderr = self._stderr
            sys.stdout = self._stdout

    @property
    def tb(self):
        # expose to caller
        return self._tb

    @staticmethod
    def _get_now_time():
        return time.strftime('%m%d_%H-%M-%S', time.localtime(time.time()))

    def _build_dir(self):
        for k, d in self._dirs.items():
            if k == 'experiment_name': continue
            if not d.exists(): d.mkdir(parents=True)

    def tb_switch(self, status):
        self._tb_switch = status
        return self

    def debug_switch(self, status):
        self._debug_switch = status
        return self

    def attach(self, experiment_name='exp', logger_name='log', override=True):
        if not override:
            experiment_name += f"_{Tracer._get_now_time()}"
        self._dirs['experiment_name'] = experiment_name
        if self._tb_switch:
            self._dirs['tensorboard'] = self._dirs['work_name'] / 'tensorboard' / f"{self._dirs['experiment_name']}_{Tracer._get_now_time()}"
        self._dirs['models'] = self._dirs['work_name'] / 'models' / self._dirs['experiment_name']
        self._dirs['checkpoint_best'] = self._dirs['models'] / 'checkpoint' / 'best'
        self._dirs['logs'] = self._dirs['work_name'] / 'logs' / self._dirs['experiment_name']

        self._build_dir()

        # edit tensorboard log_dir
        if self._tb_switch:
            self._tb = SummaryWriter(log_dir=self._dirs['tensorboard'])
            print(f"Start Tensorboard ... [tensorboard --port=6006 --logdir {self._dirs['tensorboard']}]")

        # new log file
        logger_name = self._dirs['logs'] / logger_name
        if logger_name.exists(): logger_name = f'{logger_name}_{self._get_now_time()}'

        if not self._debug_switch: self._start_log(logger_name)
        return self

    def store(self, component):
        if isinstance(component, Config):
            component.save(self._dirs['models'] / f'{Tracer.CONFIG_NAME}.json')
        if isinstance(component, Model):
            component.save(self._dirs['checkpoint_best'].parent, self._dirs['models'] / f'{Tracer.ARCH_NAME}.txt')

    def load(self, component):
        if isinstance(component, Model):
            return component.load(self._dirs['work_name'] / 'models')
