# -*- coding: utf-8 -*-
# Date: 2020/3/17 14:21

"""
some command args
"""
__author__ = 'tianyu'

from argparse import ArgumentParser


class Parser(ArgumentParser):
    def __init__(self, args_name='network'):
        self.work_name = args_name
        super(Parser, self).__init__(description=f"PyTorch implementation of {args_name}")
        self._add_default()

    def _add_default(self):
        self.add_argument('--dataset', '-ds', type=str, default='', help='the dataset for code')
        self.add_argument('-bs', dest='batch_size', type=int, default=1, help='batch size of data loader')
        self.add_argument('-j', dest='workers', type=int, default=2, help='the number of worker of data loader')
        self.add_argument('-lr', type=float, default=0.01, help='learning rate')
        self.add_argument('-wd', dest='weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
        self.add_argument('-mmt', dest='momentum', default=0.9, type=float, help='momentum')
        self.add_argument('-dp', dest='dropout', type=float, default=0.5, help='dropout')
        self.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
        self.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
        self.add_argument('-gpu', type=str, default='0', help='the number of gpu id')
        self.add_argument('--exp_suffix', type=str, default='', help='some extensional information for experiment file name')
        self.add_argument('--ext', dest='extension', type=str, default='', help='some extensional information, as flag')
        self.add_argument('--resume', default='', type=str, nargs='+', metavar='PATH', help='file name which is leaf file rather than complete path to need checkpoint don\'t contain *.pth.tar')
        self.add_argument('-eval', dest='evaluate', action='store_true', help='evaluate model on validation set')
        self.add_argument('--deterministic', action='store_true', help='fix pytorch framework seed to recurrent result')
        self.add_argument('--adjust_lr', action='store_true', help='ajust learning rate')

        # ========================= Monitor Configs ==========================
        self.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
        self.add_argument('--logger_name', '-lname', default='log.txt', type=str, help='logger name')
        self.add_argument('--work_dir', '-wdir', default='', type=str, help='workspace directory')
        self.add_argument('--clean_up', default=5, type=int, help='save top-k best checkpoint')
        self.add_argument('--debug', action='store_true', help='open debug, setting workers of dataloaer 1')
        self.add_argument('--p_bar', action='store_true', help='open process bar')
        self.add_argument('--no_tb', action='store_false', help='close tensorboard visualization')
        self.add_argument('--nowtime_exp', '-orz', action='store_false', help='automatically add nowtime as the postfix of experiment directory')


if __name__ == '__main__':
    p = Parser()
    args = p.parse_args()
    print(args)
