import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import albumentations
import cv2
from PIL import Image

defult_train_trms = albumentations.Compose([
            Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.25),
            albumentations.RandomContrast(limit=0.2, p=0.25),
            albumentations.ImageCompression(quality_lower=99, quality_upper=100),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.25),
            albumentations.ShiftScaleRotate(shift_limit=0.065, scale_limit=0.1, rotate_limit=360, border_mode=0, p=0.25),
            Cutout(max_h_size=int(512 * 0.25), max_w_size=int(512 * 0.25), num_holes=1, p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

defult_val_trms = albumentations.Compose([
            Resize(512, 512),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class TrainDataset(Dataset):
    def __init__(self, df, TRAIN_PATH, transform=None):
        self.df = df
        self.TRAIN_PATH = TRAIN_PATH
        self.file_names = df['image'].values
        self.labels = df['classes'].values
        self.labels_s = df['classes_species'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        label_species = torch.tensor(self.labels_s[idx]).long()
        return {'image':image, "label":label,"species":label_species}
    

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return {'image':image,'file_name':file_name}
    
def GetTrainDataLoader(folds,fold,train_transforms,val_transforms,batch_size,num_workers,data_root_path):
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    
    train_dataset = TrainDataset(train_folds,data_root_path, transform=train_transforms)
    valid_dataset = TrainDataset(valid_folds, data_root_path,transform=val_transforms)
    train_dataset_emb = TrainDataset(train_folds,data_root_path, transform=val_transforms)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    train_loader_emb = DataLoader(train_dataset_emb, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    
    return {"train_loader":train_loader,"train_loader_emb":train_loader_emb,"valid_loader":valid_loader}
    
