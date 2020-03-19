# -*- coding: utf-8 -*-
# Date: 2020/3/18 23:12

"""
module description
"""
__author__ = 'tianyu'

import sys
import os

apd = os.path.dirname(os.path.dirname(os.getcwd()))
print('add',apd)
sys.path.append(apd)
print(sys.path)
from torchfurnace import Tracer

from pathlib import Path

# tracer = Tracer(Path(r'.'), 'mine_network')
print(__file__)