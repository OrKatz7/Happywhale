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
    def __init__(self, df, TRAIN_PATH, transform=None,crop_p=None,crop_csv_path=None):
        self.df = df
        self.TRAIN_PATH = TRAIN_PATH
        self.file_names = df['image'].values
        self.labels = df['classes'].values
        self.labels_s = df['classes_species'].values
        self.transform = transform
        
        if crop_csv_path is not None and crop_p is not None:
            self.crop_df = pd.read_csv(crop_csv_path)
            self.crop_p = crop_p
            self.use_crop = True
        else:
            self.use_crop = False
        
    def __len__(self):
        return len(self.df)
    
    def Crop(self,image,xmin, ymin, xmax, ymax):
        offset_y = torch.rand(1)[0]/4 if self.crop_p != 1 else 0.1
        offset_x = torch.rand(1)[0]/4 if self.crop_p != 1 else 0.1
        image = image[int(max(0,ymin*(1-offset_y))):int(min(image.shape[0],ymax*(1+offset_y))),int(max(0,xmin*(1-offset_x))):int(min(image.shape[1],xmax*(1+offset_x)))]
        return image
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.use_crop and torch.rand(1)[0]<self.crop_p:
            row = self.crop_df[self.crop_df.image == file_name]
            if len(row) > 0:
                row = row.values[0]
                conf = row[4]
                if conf > 0.1:
                    xmin, ymin, xmax, ymax = row[0:4]
                    image = self.Crop(image,xmin, ymin, xmax, ymax)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        label_species = torch.tensor(self.labels_s[idx]).long()
        return {'image':image, "label":label,"species":label_species}
    

class TestDataset(Dataset):
    def __init__(self, df, TRAIN_PATH, transform=None,crop_p=None,crop_csv_path=None):
        self.df = df
        self.TRAIN_PATH = TRAIN_PATH
        self.file_names = df['image'].values
        self.transform = transform
        
        if crop_csv_path is not None and crop_p is not None:
            self.crop_df = pd.read_csv(crop_csv_path)
            self.crop_df = self.crop_df.fillna('[0]')
            self.crop_df['bbox'] = self.crop_df['bbox'].map(eval)
            self.crop_df['conf'] = self.crop_df['conf'].map(eval)
            self.crop_p = crop_p
            self.use_crop = True
        else:
            self.use_crop = False
        
    def __len__(self):
        return len(self.df)
    
    def Crop(self,image,xmin, ymin, xmax, ymax):
        offset_y = torch.rand(1)[0]/4 if self.crop_p != 1 else 0
        offset_x = torch.rand(1)[0]/4 if self.crop_p != 1 else 0
        image = image[int(max(0,ymin*(1-offset_y))):int(min(image.shape[0],ymax*(1+offset_y))),int(max(0,xmin*(1-offset_x))):int(min(image.shape[1],xmax*(1+offset_x)))]
        return image
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.use_crop and torch.rand(1)[0]<self.crop_p:
            row = self.crop_df[self.crop_df.image_id == file_name]
            conf = row['conf'].values
            if conf[0][0] > 0.2:
                bbox = row['bbox'].values 
                xmin, ymin, xmax, ymax = bbox[0][0]
                image = self.Crop(image,xmin, ymin, xmax, ymax)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return {"file_name":file_name,'image':image, "label":torch.empty([]),"species":torch.empty([])}
    
    
def GetTrainDataLoader(folds,fold,train_transforms,val_transforms,batch_size,num_workers,data_root_path,crop_p,crop_csv_path,use_crop_for_val):
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    
    train_dataset = TrainDataset(train_folds,data_root_path, transform=train_transforms,crop_p=crop_p,crop_csv_path=crop_csv_path)
    valid_dataset = TrainDataset(valid_folds, data_root_path,transform=val_transforms,crop_p=1 if use_crop_for_val else 0,crop_csv_path=crop_csv_path)
    train_dataset_emb = TrainDataset(train_folds,data_root_path, transform=val_transforms,crop_p=1 if use_crop_for_val else 0,crop_csv_path=crop_csv_path)

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
    
