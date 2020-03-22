# -*- coding: utf-8 -*-
# Date: 2020/3/18 15:46

"""
some components for tracer class
"""
__author__ = 'tianyu'

import configparser
import json
import shutil
from pathlib import Path

import torch

from .function import log


class Config(object):
    """
    https://github.com/OIdiotLin/torchtracer/blob/master/torchtracer/data/config.py
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        if isinstance(cfg, configparser.ConfigParser):
            self.content = Config.from_cfg(cfg)
        elif isinstance(cfg, str):
            self.content = Config.from_ini(cfg)
        elif isinstance(cfg, dict):
            self.content = json.dumps(Config.from_dict(cfg), indent=2)

    @staticmethod
    def from_ini(ini):
        config = configparser.ConfigParser()
        config.read_string(ini)
        return Config.from_cfg(config)

    @staticmethod
    def from_cfg(cfg):
        dic = {}
        sections = cfg.sections()
        for section in sections:
            dic_section = {}
            options = cfg.options(section)
            for option in options:
                dic_section[option] = cfg.get(section, option)
            dic[section] = dic_section
        return dic

    @staticmethod
    def from_dict(dic):
        res = {}
        # only loss function name reserved.
        if isinstance(dic, torch.nn.modules.loss._Loss):
            return dic._get_name()
        #
        if isinstance(dic, torch.optim.Optimizer):
            sub = dic.param_groups[0].copy()
            sub.pop('params')
            sub['name'] = dic.__class__.__name__
            return Config.from_dict(sub)
        for k in dic.keys():
            if type(dic[k]) in [int, float, bool, str, list]:
                res[k] = dic[k]
            elif isinstance(dic[k], (torch.optim.Optimizer,
                                     torch.nn.modules.loss._Loss)):
                res[k] = Config.from_dict(dic[k])
        return res

    def __repr__(self):
        return self.content

    def save(self, path: Path):
        path.open('w', encoding='utf-8').write(str(self.content))


class Model(object):

    def __init__(self, name, state, is_best=False):
        """
        usage
        :param name: name if save mode
                     experiment_name/file_name*_best.pth.tar if load mode
        :param state:
                save func: key: 'epoch' 'state_dict' 'best_acc1' 'optimizer' 'arch'
                load func: key: 'model' 'optim'
        :param is_best:
        """
        self._name = f"{name}"
        self._state = state
        self._is_best = is_best

    def save(self, pre_ckp_path: Path, arch_path: Path):
        # save network architecture
        arch_path.open('w', encoding='utf-8').write(self._state['arch'])

        # save checkpoint
        torch.save(self._state, pre_ckp_path / self._name)
        if self._is_best:
            import re
            # ckp_name = f"{model.__class__.__name__}_Epk{self._state['epoch'] + 1}_Acc{self._state['best_acc1']:.2f}{postfix}.pth.tar"
            model_name = re.split("[.,_]", self._name, 1)[0]
            best_name = self._name.replace('.pth.tar', '_best.pth.tar') \
                .replace(model_name, f"{model_name}_Epk{self._state['epoch']}_Acc{self._state['best_acc1']:.2f}")
            shutil.copyfile(Path(pre_ckp_path / self._name), pre_ckp_path / 'best' / best_name)

    def load(self, pre_path: Path) -> dict:
        # pre_path: located in work_name/models
        exp_name, file_name = self._name.split('/')
        file_path = pre_path / exp_name / 'checkpoint' / 'best' / file_name
        ret = {'start_epoch': -1, 'best_acc1': -1}
        if file_path.is_file():
            log("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path)
            ret['start_epoch'] = checkpoint['epoch']
            ret['best_acc1'] = checkpoint['best_acc1']
            self._state['model'].load_state_dict(checkpoint['state_dict'])
            self._state['optim'].load_state_dict(checkpoint['optimizer'])
            log("=> loaded checkpoint '{}' (epoch {} Acc@1 {})"
                .format(file_path, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            assert False, "=> no checkpoint found at '{}'".format(file_path)
        return ret

        # return load_checkpoint(path / self._name.replace('.pth.tar', '_best.pth.tar'), self._state)
