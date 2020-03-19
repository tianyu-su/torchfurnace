# -*- coding: utf-8 -*-
# Date: 2020/3/18 23:12

"""
module description
"""
__author__ = 'tianyu'

from torchfurnace import Tracer

from pathlib import Path

tracer = Tracer(Path(r'E:\OneDrive - stu.ouc.edu.cn\ToolChains\torchfurnace\torchfurnace\tests'), 'mine_network')
tracer.tb