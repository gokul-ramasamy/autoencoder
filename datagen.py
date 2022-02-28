#Data Generator code for a given set of chest x-rays

import os
import glob
from re import I
import torch
from skimage import io, transform, img_as_ubyte
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import random
from json_parser import TRAIN_TEXT, TEST_TEXT, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, INPUT_SIZE
from PIL import Image


#Amara's equalisation code
from skimage import exposure
import numpy as np
def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)

#Adaptive equalisation code
from skimage.exposure import equalize_adapthist
def adaptive_equalisation(img):
    return equalize_adapthist(img)

#Contrast stretching code
from skimage.exposure import rescale_intensity
def contrast_stretch(img):
    return rescale_intensity(img, in_range=(0,255), out_range=(0,255))


#Reading the paths from train files
with open(TRAIN_TEXT, 'r') as f:
    data = f.readlines()
    train_paths = list()
    for i in data:
        train_paths.append(i)
    train_paths = [i[:-1] for i in train_paths]

#Reading the paths from test files
with open(TEST_TEXT, 'r') as f:
    data = f.readlines()
    test_paths = list()
    for i in data:
        test_paths.append(i)
    test_paths = [i[:-1] for i in test_paths]

train_paths = train_paths
test_paths = test_paths

# class ChestXDataset(Dataset):
#     """ Dataset Class"""
#     def __init__(self, data_paths, transform=None):
#         self.all_paths = data_paths
#         self.transform = transform
    
#     #Overriding the __len__ method as the number of images in the root folder directory
#     def __len__(self):
#         return len(self.all_paths)

#     #Overriding the __getitem__ method to get the images in the root folder directory
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image_name = self.all_paths[idx]
#         try:
#             image = io.imread(image_name) ###Looks like image should be a PIL image and not skimage
#         except:
#             image = np.zeros((256,256))
#             print(image_name)
        
#         sample = {'image':image}
#         if self.transform:
#             sample = self.transform(sample)

#         return sample

class ChestXDataset(Dataset):
    """ Dataset Class"""
    def __init__(self, data_paths, transform=None):
        self.all_paths = data_paths
        self.transform = transform
    
    #Overriding the __len__ method as the number of images in the root folder directory
    def __len__(self):
        return len(self.all_paths)

    #Overriding the __getitem__ method to get the images in the root folder directory
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_paths[idx]

            
        image = io.imread(image_name)
        if image.dtype != 'uint8': #Checking whether the image is 8 bit, if not do the following
        #Normalise the image to [0,1] and then continue with the 8 bit conversion (multiply by 255)
            normalized = (image.astype(np.uint16) - image.min()) * 255.0 / (image.max() - image.min()) #Converts to 0-255 scale
            image = normalized.astype(np.uint8) 

        sample = Image.fromarray(image)
        if self.transform:
            sample = self.transform(sample)

        return sample



class Rescale_Norm(object):
    """Rescaling and Normalizing the image"""
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
    
    def __call__(self, sample):
        image = sample['image']

        new_h, new_w = self.output_size
        # image = (image/256).astype('uint8') #Conversion of 16-bit image into an 8-bit image
        image = img_as_ubyte(image) #Conversion of 16-bit image into an 8-bit image
        # image = image/255 #Since transform.resize normalises the image -> this is not required
        img = transform.resize(image, (new_h, new_w), anti_aliasing=True) #transform.resize normalizes the image as well

        return {'image':img}

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']

        # image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image)}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top:top+new_h, left:left+new_w]
        return {'image':image}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        if random.random() > 0.5:
            return {'image':torch.flip(image,[1])}
        else:
            return {'image':image}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        if random.random() > 0.5:
            return {'image':torch.flip(image,[0])}
        else:
            return {'image':image}



#Loading the images
train_transformed_dataset = ChestXDataset(data_paths=train_paths,
                        transform = transforms.Compose([
                                                        transforms.Resize((282,282)),
                                                        transforms.CenterCrop(256),
                                                        transforms.ToTensor(),
                                                        transforms.RandomRotation(10),
                                                        transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
                                                        transforms.Normalize((0.5033,),(0.2887,)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomVerticalFlip()
                                                        ])
                        )

test_transformed_dataset = ChestXDataset(data_paths=test_paths,
                        transform = transforms.Compose([
                                                        transforms.Resize((282,282)),
                                                        transforms.CenterCrop(256),
                                                        transforms.ToTensor(),
                                                        transforms.RandomRotation(10),
                                                        transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
                                                        transforms.Normalize((0.5033,),(0.2887,)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomVerticalFlip()
                                                        ])
                        )


#DataLoader
train_dataloader = DataLoader(train_transformed_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

print('>> Dataloader successful..')



