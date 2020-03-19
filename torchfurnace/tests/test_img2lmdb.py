# -*- coding: utf-8 -*-
# Date: 2020/3/19 0:22

"""
module description
"""
__author__ = 'tianyu'

from torchfurnace import ImageLMDB, ImageFolderLMDB
import matplotlib.pyplot as plt


plt.imshow(dataset[0][0]) # PIL object
plt.axis('off')
plt.show()