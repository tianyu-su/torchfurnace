# -*- coding: utf-8 -*-
# Date: 2020/3/19 0:22

"""
module description
"""
__author__ = 'tianyu'

import sys
from pathlib import Path

sys.path.append(Path(__file__).absolute().parent.parent.parent)

# from torchfurnace import ImageLMDB, ImageFolderLMDB

# import matplotlib.pyplot as plt
#
#
# plt.imshow(dataset[0][0]) # PIL object
# plt.axis('off')
# plt.show()
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print('hhh',BASE_DIR)
print(__file__)
from pathlib import Path

print(Path(__file__).absolute())
print(Path(__file__).absolute().parent.parent.parent)
