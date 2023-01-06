# libraries pytorch
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# classic
import numpy as np
from PIL import Image
import os, random, glob


##################################
# Segmentation Loader
##################################	

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SlimeDataSet():

    def __init__(self, path_folder, transform_train, transform_test, pct_train_set=.8, shuffle_dataset=True):
        # setup path+names
        filenames_tp = sorted(set(glob.glob(path_folder + "*.jpg")))
        self.filenames = [f[:-4] for f in filenames_tp]  # remove ".jpg"
        # setup the transformation
        self.transform_train = transform_train
        self.transform_test = transform_test
        # split train/test set
        N = len(self.filenames)
        indices = list(range(N))
        split = int(np.floor(pct_train_set * N))
        if shuffle_dataset:
            random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # get image
        filename = self.filenames[idx]
        with open(filename + '.jpg', 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(filename + '.png', 'rb') as f:
            mask = Image.open(f).convert('P')
        # apply transformation
        if idx in self.train_sampler:
            seed = np.random.randint(0, 40000)  # use the same seed to img and mask
            init_seed(seed)
            X = self.transform_train(image)
            init_seed(seed)
            y = self.transform_train(mask).squeeze(0).long()
        else:
            X = self.transform_test(image)
            y = self.transform_test(mask).squeeze(0).long()

        return X, y

    def __len__(self):
        return len(self.filenames)


class SlimeDataSet_prediction():

    def __init__(self, filenames, transform_test, test_sampler):
        # setup path+names
        self.filenames = filenames
        self.transform_test = transform_test
        self.test_sampler = test_sampler

    def __getitem__(self, idx):
        # get image
        filename = self.filenames[idx]
        with open(filename + '.jpg', 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(filename + '.png', 'rb') as f:
            mask = Image.open(f).convert('P')
        # apply transformation
        X = self.transform_test(image)
        y = self.transform_test(mask).squeeze(0).long()
        # short filename
        filename_short = filename.split('/')[-1]
        # was it in the training set?
        inTestSet = (idx in self.test_sampler)

        return X, y, filename_short, inTestSet

    def __len__(self):
        return len(self.filenames)
