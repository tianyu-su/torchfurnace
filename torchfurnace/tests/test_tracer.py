# -*- coding: utf-8 -*-
# Date: 2020/3/18 23:12

"""
module description
"""
__author__ = 'tianyu'

import sys
from pathlib import Path
print('add path:',str(Path(__file__).absolute().parent.parent.parent))

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))


from torchfurnace import Tracer

from pathlib import Path

# tracer = Tracer(Path(r'.'), 'mine_network')
print(__file__)
