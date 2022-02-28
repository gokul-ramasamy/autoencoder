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



