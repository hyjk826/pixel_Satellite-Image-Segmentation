import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

category_names = ['Background', 'building']

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SatelliteDataset(Dataset):
    def __init__(self, 
                 img_root, 
                 anno_root, 
                 infer=False, 
                 ext=[".png"], 
                 train_transform=None,
                 infer_transform=None):
        
        self.img_root = img_root
        self.anno_root = anno_root
        self.infer = infer
        self.ext = ext
        self.img_name_list = []
        for img_name in os.listdir(img_root):
            _, extension = os.path.splitext(img_name)
            if extension in self.ext:
                self.img_name_list.append(img_name)
                
        self.train_transform = train_transform
        self.infer_transform = infer_transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_path = os.path.join(self.img_root, img_name)
        image = cv2.imread(img_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.infer_transform:
                image = self.infer_transform(image=image)['image']
            return image
        
        mask_path = os.path.join(self.anno_root, img_name)
        mask = cv2.imread(mask_path, 0)
        if self.train_transform:
            augmented = self.train_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask