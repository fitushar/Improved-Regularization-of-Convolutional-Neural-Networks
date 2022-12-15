"""
Copyright <2022> <ECE661 Team, Duke University>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import os.path
import numpy as np
import sys
import torch

from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data import Dataset as VisionDataset
from tools.utils import check_integrity, download_and_extract_archive

import shutil


class CIFAR10:
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root, mode='train', download=True, transform=None, target_transform=None):
        if mode in ['train', 'val']:
            self.url = "https://www.dropbox.com/s/s8orza214q45b23/cifar10_trainval_F22.zip?dl=1"
            self.filename = "cifar10_trainval_F22.zip"
        else:
            self.url = "https://www.dropbox.com/s/mine6hvnxsi3dio/cifar10_test_F22.zip?dl=1"
            self.filename = "cifar10_test_F22.zip"
            
        self.transform = transform
        self.target_transform = target_transform

        self.root = root
        if download:
            self.download()
        self.mode = mode
        
        self.data = []
        self.targets = []

        if self.mode == 'train':
            self.data = np.load(os.path.join(root, "train-images-batch.npy"))
            self.targets = np.asarray(
                np.load(os.path.join(root, "train-labels-batch.npy")), dtype=np.int
            )
        elif self.mode == 'val':
            self.data = np.load(os.path.join(root, "val-images-batch.npy"))
            self.targets = np.asarray(
                np.load(os.path.join(root, "val-labels-batch.npy")), dtype=np.int
            )
        elif self.mode == 'test':
            self.data = np.load(os.path.join(root, "test-images-batch.npy"))            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.mode != 'test':
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        elif self.mode == 'test':
            img = self.data[index]
            
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
                
            return img
        
    def download(self):
        try:
            download_and_extract_archive(self.url, self.root, filename=self.filename)
        except Exception as e:
            print("Interrupted during dataset downloading. "
                  "Cleaning up...")
            # Clean up
            cwd = os.getcwd()
            rm_path = os.path.join(cwd, self.root, self.filename.split('.')[0])
            shutil.rmtree(rm_path)
            raise e

        print('Files already downloaded and verified')

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return f"Split: {self.mode}"
