import torch.utils.data as data
import torch
import h5py
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')

        self.GT = np.asarray(dataset['GT'])
        self.LRHSI = np.asarray(dataset['LRHSI'])
        self.RGB = np.asarray(dataset['RGB'])

        self.GT = self.GT.transpose((0, 3, 1, 2))
        self.LRHSI = self.LRHSI.transpose((0, 3, 1, 2))
        self.RGB = self.RGB.transpose((0, 3, 1, 2))

    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.GT[index, :, :, :]).float(), \
               torch.from_numpy(self.LRHSI[index, :, :, :]).float(), \
               torch.from_numpy(self.RGB[index, :, :, :]).float()


                   #####必要函数
    def __len__(self):
        return self.GT.shape[0]
