# -*- coding: utf-8 -*-
# Date: 2020/3/17 15:35

"""
save picture to LMDB
"""
__author__ = 'tianyu'

import os
import os.path as osp

import lmdb
import pyarrow as pa
import six
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


class ImageFolderLMDB(data.Dataset):
    """
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    """

    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    @staticmethod
    def folder2lmdb(dpath, name="train", write_frequency=5000, num_workers=16, lmdb_size=20480000):
        """
        The dpath has this structure of directory as follow
        |- dpath
        |------- train
        |------------- dog
        |----------------- 001.jpg
        |------------- cat
        |----------------- 001.jpg
        |-------- val
        |------------- dog
        |----------------- 001.jpg
        |------------- cat
        |----------------- 001.jpg

        :param name: train or val
        :param lmdb_size: the size of lmdb (Byte)
        """
        directory = osp.expanduser(osp.join(dpath, name))
        print("Loading dataset from %s" % directory)

        dataset = ImageFolder(directory, loader=raw_reader)

        data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

        lmdb_path = osp.join(dpath, "%s.lmdb" % name)
        isdir = os.path.isdir(lmdb_path)

        print("Generate LMDB to %s" % lmdb_path)
        db = lmdb.open(lmdb_path, subdir=isdir,
                       map_size=lmdb_size, readonly=False,
                       meminit=False, map_async=True)

        print(len(dataset), len(data_loader))
        txn = db.begin(write=True)
        for idx, data in enumerate(data_loader):
            # print(type(data), data)
            image, label = data[0]
            txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(data_loader)))
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', dumps_pyarrow(keys))
            txn.put(b'__len__', dumps_pyarrow(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()


class ImageLMDB(object):

    @staticmethod
    def _dataset4store(data_mapping):
        """ you can override it for your requirements.
        For example, you want to store over one flow images,
        and you can put each list of flow picture path,into data_mapping['pic_path],
        for instance, data_mapping['pic_path].append(['flow1.jpg','flow2.jpg'])
        """

        class InnerDataset(data.Dataset):
            def __init__(self, data_mapping):
                self.keys = data_mapping['key']
                self.pics = data_mapping['pic_path']

            def __getitem__(self, item):
                return self.keys[item], raw_reader(self.pics[item])

            def __len__(self):
                return len(self.keys)

        return InnerDataset(data_mapping)

    @staticmethod
    def store(data_mapping, dpath, lmdb_size, num_workers=16, write_frequency=5000):
        """
        store each pictures in the `dada_mapping` with small size to LMDB
        :param data_mapping: {'key':[],'pic_path':[]}
        :param dpath: direcotry path path to save lmdb file
        :param lmdb_size: approximate space
        :param num_workers: processing worker
        :param write_frequency: frequency of writing file
        :return:
        """
        data_loader = DataLoader(ImageLMDB._dataset4store(data_mapping), num_workers=num_workers, collate_fn=lambda x: x)
        data_loader = tqdm(data_loader, desc='Processing')
        print(f"Generate LMDB to {dpath}")
        db = lmdb.open(dpath, map_size=lmdb_size, readonly=False,
                       meminit=False, map_async=True)
        print(f"total sample: {len(data_loader):,}")
        txn = db.begin(write=True)
        for idx, data in enumerate(data_loader):
            key, image = data[0]
            txn.put(u'{}'.format(key).encode('ascii'), dumps_pyarrow(image))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(data_loader)))
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        with db.begin(write=True) as txn:
            txn.put(b'__len__', dumps_pyarrow(len(data_loader)))

        print("Flushing database ...")
        db.sync()
        db.close()

    @staticmethod
    def _post4store(unpacked_val):
        """ you can override it for your requirements.
         For example, you can return a list of PIL object whose raw data is contained in unpacked_val
         when you want to read over one flow images.

        :param unpacked_val: the raw data needed to convert to binary format for value of LMDB
        :return:
        """
        # load single image
        buf = six.BytesIO()
        buf.write(unpacked_val)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    @staticmethod
    def read(db_path, key):
        """
        read image(s) form LMDB by key
        :param db_path: LMDB location
        :param key: the key of value
        :return: PIL image(s) which can continue to be processed with pytorch transform
        """
        env = lmdb.open(db_path, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            byteflow = txn.get(u'{}'.format(key).encode('ascii'))
        unpacked = pa.deserialize(byteflow)

        return ImageLMDB._post4store(unpacked)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=0)

    args = parser.parse_args()
    args.folder = r'E:\OneDrive - stu.ouc.edu.cn\ToolChains\torchfurnace\torchfurnace\tests\imgs'

    ImageFolderLMDB.folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
