# libraries pytorch
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# classic
import numpy as np
from PIL import Image
from PIL import ImageOps
import os, random, glob


##################################
# Segmentation Loader
##################################

def init_seed(seed):
    ''' Initializes seed for Torch and NumPy
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SlimeDataSet():

    def __init__(self, path_folder, transform_train, transform_test, pct_train_set=.8, shuffle_dataset=True, box_or_slime=1): 
        # setup path+names
        filenames_tp = sorted(set(glob.glob(path_folder + "*.jpg")))
        
        self.box_or_slime = box_or_slime
        # adjust file naming conventions based on slimenet or boxnet data
        if box_or_slime == 2:  # slimenet mode
            self.crop_amount = -9
            self.image_suffix = "_crop.jpg"
            self.mask_suffix = "_crop.png"
            # for slimenet: load in circular mask to crop out just the petri dish
            with open('resources/petri_template.png', 'rb') as f:
                petri_mask = Image.open(f).convert('P')
            toTensor = torchvision.transforms.ToTensor()
            self.petri_mask = toTensor(petri_mask)
        
        else:  # boxnet mode
            self.crop_amount = -12
            self.image_suffix = "_resized.jpg"
            self.mask_suffix = "_boxMask_resized.png"

        self.filenames = [f[:self.crop_amount] for f in filenames_tp]  # remove ".jpg"
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
        with open(filename + self.image_suffix, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(filename + self.mask_suffix, 'rb') as f:
            if self.box_or_slime == 2:
                mask = Image.open(f).convert('P')
            else:
                mask = ImageOps.invert(Image.open(f)).convert('P')
        # apply transformation
        if idx in self.train_sampler:
            seed = np.random.randint(0, 40000)  # use the same seed to img and mask
            init_seed(seed)
            X = self.transform_train(image)
            init_seed(seed)
            y = torch.round(self.transform_train(mask)).long()
            if self.box_or_slime == 2:
                y = 1 - y
                # element-wise multiplication sets every pixel outside the petri dish equal to zero
                X *= self.petri_mask
        else:
            X = self.transform_test(image)
            y = torch.round(self.transform_test(mask)).long()
            if self.box_or_slime == 2:
                y = 1 - y
                X *= self.petri_mask

        return X, y

    def __len__(self):
        return len(self.filenames)


class SlimeDataSet_prediction():

    def __init__(self, filenames, transform_test, test_sampler, box_or_slime):
        # setup path+names
        
        # adjust file naming conventions based on slimenet or boxnet data
        self.box_or_slime = box_or_slime
        if box_or_slime == 2:  # slimenet mode
            self.crop_amount = 9
            self.image_suffix = "_crop.jpg"
            self.mask_suffix = "_crop.png"
            with open('resources/petri_template.png', 'rb') as f:
                petri_mask = Image.open(f).convert('P')
            toTensor = torchvision.transforms.ToTensor()
            self.petri_mask = toTensor(petri_mask)

        else: # boxnet mode
            self.crop_amount = 12
            self.image_suffix = "_resized.jpg"
            self.mask_suffix = "_boxMask_resized.png"
        
        self.filenames = filenames
        self.transform_test = transform_test
        self.test_sampler = test_sampler
        

    def __getitem__(self, idx):
        # get image
        filename = self.filenames[idx]
        with open(filename + self.image_suffix, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(filename + self.mask_suffix, 'rb') as f:
            if self.box_or_slime == 2:
                mask = Image.open(f).convert('P')
            else:
                mask = ImageOps.invert(Image.open(f)).convert('P')
        # apply transformation
        X = self.transform_test(image)
        y = torch.round(self.transform_test(mask)).long()
        # short filename
        filename_short = filename.split('/')[-1]
        # was it in the training set?
        inTestSet = (idx in self.test_sampler)
        if self.box_or_slime == 2: # set pixels outside the petri dish equal to zero
            X *= self.petri_mask

        return X, y, filename_short, inTestSet

    def __len__(self):
        return len(self.filenames)
