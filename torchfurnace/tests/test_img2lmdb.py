# -*- coding: utf-8 -*-
# Date: 2020/3/19 0:22

"""
test image2lmdb module
"""
__author__ = 'tianyu'

import sys
from pathlib import Path

print('add path:', str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(sys.path)
print(f'Testing {Path(__file__)}')

from torchfurnace import ImageLMDB, ImageFolderLMDB
from torchfurnace.utils.decorator import test_function


@test_function
def test_imagefolderldmb_folder2lmdb():
    import platform
    ImageFolderLMDB.folder2lmdb('demo_dataset', name='train', num_workers=0 if platform.system() == 'Windows' else 16)


@test_function
def test_test_imagefolderldmb_class():
    from torch.utils.data import DataLoader
    _set = ImageFolderLMDB('demo_dataset/train.lmdb')
    loader = DataLoader(_set, batch_size=1, collate_fn=lambda x: x)

    for idx, data in enumerate(loader):
        img, target = data[0]
        print(f'[{idx + 1}/{len(loader)}]  image:{img.size} target:{target}')
        import platform
        if platform.system() == 'Windows':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.title(f'Class: {target}')
            plt.axis('off')
            plt.show()


@test_function
def test_ImageLMDB_sotre():
    data_mapping = {
        'key': [f"{k}_{no:04}" for k in ['dog', 'cat'] for no in range(1, 4)],
        'pic_path': [f"demo_imgs/{k}{no:03}.jpg" for k in ['d', 'c'] for no in range(1, 4)]
    }
    import platform

    ImageLMDB.store(data_mapping, 'demo_imgs_db', 4800000, num_workers=0 if platform.system() == 'Windows' else 16)


@test_function
def test_ImageLMDB_read():
    data_mapping = {
        'key': [f"{k}_{no:04}" for k in ['dog', 'cat'] for no in range(1, 4)],
        'pic_path': [f"demo_imgs/{k}{no:03}.jpg" for k in ['d', 'c'] for no in range(1, 4)]
    }

    for key in data_mapping['key']:
        img = ImageLMDB.read('demo_imgs_db', key)
        import platform
        if platform.system() == 'Windows':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    test_imagefolderldmb_folder2lmdb()
    test_test_imagefolderldmb_class()

    test_ImageLMDB_sotre()
    test_ImageLMDB_read()
