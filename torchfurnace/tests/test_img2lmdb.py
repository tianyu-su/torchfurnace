# -*- coding: utf-8 -*-
# Date: 2020/3/19 0:22

"""
module description
"""
__author__ = 'tianyu'

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from torchfurnace import ImageLMDB, ImageFolderLMDB

# import matplotlib.pyplot as plt
#
#
# plt.imshow(dataset[0][0]) # PIL object
# plt.axis('off')
# plt.show()

# print(os.path.dirname(os.getcwd()))
print(__file__)
