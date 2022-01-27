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
        try:
            image = io.imread(image_name)
        except:
            image = np.zeros((256,256))
            print(image_name)
        sample = {'image':image}
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


train_transformed_dataset = ChestXDataset(data_paths=train_paths,
                                    transform=transforms.Compose([Rescale_Norm(INPUT_SIZE), 
                                                                ToTensor()
                                                                # RandomHorizontalFlip(),
                                                                # RandomVerticalFlip()
                                                                ]))
test_transformed_dataset = ChestXDataset(data_paths=test_paths,
                                    transform=transforms.Compose([Rescale_Norm(INPUT_SIZE),
                                                                ToTensor()
                                                                # RandomHorizontalFlip(),
                                                                # RandomVerticalFlip()
                                                                ]))



#DataLoader
train_dataloader = DataLoader(train_transformed_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

print('>> Dataloader successful..')



