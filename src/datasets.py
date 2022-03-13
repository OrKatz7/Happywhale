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
from tqdm import tqdm
from torch.utils.data.sampler import Sampler


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

class StepSampler(Sampler):
    def __init__(self, data_source, epochs: int = 25):
        super().__init__(data_source)
        self.indices0 = np.where(data_source.count > 5)[0]
        self.indices1 = np.where(data_source.count > 3)[0]
        self.indices2 = np.where(data_source.count > 2)[0]
        self.indices3 = np.where(data_source.count > 1)[0]
        self.current_epoch: int = 0
        self.epochs: int = epochs
        self.indices = self.indices0

    def __iter__(self):
        if self.current_epoch < 5:#8:
            indices = np.random.permutation(np.hstack((self.indices0)))
        elif self.current_epoch < 10: #13
            indices = np.random.permutation(np.hstack((self.indices0 , self.indices1)))
        elif self.current_epoch < 18: #20
            indices = np.random.permutation(np.hstack((self.indices0 , self.indices1, self.indices2)))
        else:
            indices = np.random.permutation(np.hstack((self.indices0 , self.indices1, self.indices2,self.indices3)))
        # indices = list(set(indices.tolist()))
        self.indices = indices
        return iter(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        print(f"sampler set epoch {epoch}")

class TrainDataset(Dataset):
    def __init__(self, df, TRAIN_PATH, transform=None,crop_p=None,crop_csv_path=None,crop_backfin_csv_path = None,mode='train',load_all=False,
                 new_targets=False,hide=0):
        self.df = df.copy()
        self.TRAIN_PATH = TRAIN_PATH
        self.file_names = df['image'].values
        self.labels = df['classes'].values
        self.labels_s = df['classes_species'].values
        self.folds=df['fold'].values
        self.count=df['count'].values
        if hide>0:
            df.not_seen=(df.num_images>hide) & (df.not_seen)
        self.not_seen=df['not_seen'].values
        self.transform = transform
        self.mode = mode
        self.new_targets=new_targets
        if crop_csv_path is not None and crop_p is not None:
            self.crop_df = pd.read_csv(crop_csv_path)
            self.crop_backfin_df = pd.read_csv(crop_backfin_csv_path)
            self.crop_p = crop_p
            self.use_crop = True
        else:
            self.use_crop = False
        self.x,self.y,self.y_species,self.y_folds,self.y_not_seen = [],[],[],[],[]
        if load_all:
            self.load_all()
        self.load_all = load_all
            
    def load_all(self):
        for idx in tqdm(range(len(self.file_names))):
            file_name = self.file_names[idx]
            file_path = f'{self.TRAIN_PATH}/{file_name}'
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            row = self.crop_backfin_df[self.crop_backfin_df.image == file_name]
            if len(row) == 0:
                row = self.crop_df[self.crop_df.image == file_name]
            if len(row) > 0:
                row = row.values[0]
                conf = row[4]
                if conf > 0:
                    xmin, ymin, xmax, ymax = row[0:4].astype(int)
                    image = image[ymin:ymax,xmin:xmax]
            self.x.append(image)
            self.y.append(torch.tensor(self.labels[idx]).long())
            self.y_species.append(torch.tensor(self.labels_s[idx]).long())
            self.y_folds.append(torch.tensor(self.folds[idx]).long())
            self.y_not_seen.append(torch.tensor(self.not_seen[idx]).long())
    def __len__(self):
        return len(self.df)
    
    def Crop(self,image,xmin, ymin, xmax, ymax):
        offset_y = torch.rand(1)[0]/5 if self.crop_p != 1 else 0
        offset_x = torch.rand(1)[0]/5 if self.crop_p != 1 else 0
        image = image[int(max(0,ymin*(1-offset_y))):int(min(image.shape[0],ymax*(1+offset_y))),int(max(0,xmin*(1-offset_x))):int(min(image.shape[1],xmax*(1+offset_x)))]
        return image
    
    def __getitem__(self, idx):
        if not self.load_all:
            file_name = self.file_names[idx]
            file_path = f'{self.TRAIN_PATH}/{file_name}'
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if (self.use_crop and torch.rand(1)[0]<self.crop_p) or self.mode=='test':
                # print("crop")
                row = self.crop_backfin_df[self.crop_backfin_df.image == file_name]
                if len(row) == 0:
                    row = self.crop_df[self.crop_df.image == file_name]
                if len(row) > 0:
                    row = row.values[0]
                    conf = row[4]
                    if conf > 0:
                        xmin, ymin, xmax, ymax = row[0:4]
                        image = self.Crop(image,xmin, ymin, xmax, ymax)
            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                print(file_path)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            label = torch.tensor(self.labels[idx]).long()
            label_species = torch.tensor(self.labels_s[idx]).long()
            fold=torch.tensor(self.folds[idx]).long()
            not_seen=torch.tensor(self.not_seen[idx]).long()
        else:
            image = self.x[idx]
            label = torch.tensor([-1]).long() if (self.new_targets and self.y_not_seen[idx]==0) else self.y[idx]
            label_species = self.y_species[idx]
            fold = self.y_folds[idx]
            not_seen=self.y_not_seen[idx]

            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                image = cv2.imread(self.file_names[idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
        return {'image':image, "label":label,"species":label_species,"fold":fold,"not_seen":not_seen}
    

class TestDataset(Dataset):
    def __init__(self, df, TRAIN_PATH, transform=None,crop_p=None,crop_csv_path=None,crop_backfin_csv_path = None):
        self.df = df
        self.TRAIN_PATH = TRAIN_PATH
        self.file_names = df['image'].values
        self.transform = transform

        if crop_csv_path is not None and crop_p is not None:
            self.crop_df = pd.read_csv(crop_csv_path)
            self.crop_backfin_df = pd.read_csv(crop_backfin_csv_path)
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
        if True:
            row = self.crop_backfin_df[self.crop_backfin_df.image == file_name]
            if len(row) == 0:
                row = self.crop_df[self.crop_df.image == file_name]
            if len(row) > 0:
                row = row.values[0]
                conf = row[4]
                if conf > 0:
                    xmin, ymin, xmax, ymax = row[0:4]
                    image = self.Crop(image,xmin, ymin, xmax, ymax)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return {"file_name":file_name,'image':image, "label":torch.empty([]),"species":torch.empty([])}
    
    
def GetTrainDataLoader(folds,fold,train_transforms,val_transforms,batch_size,num_workers,data_root_path,
                       crop_p,crop_csv_path,crop_backfin_csv_path,use_crop_for_val,
                       use_sampler=True,epochs=25,
                       singles_in_fold=False,no_single_val=True,singles_in_train=True,min_num_in_train=1,train_not_seen=False):
    # if train_not_seen:
    #     folds=folds.copy()
    #     folds.not_seen=(folds.num_images>min_num_in_train).to(torch.long)
    if not singles_in_train:
        trn_idx = folds[(folds['fold'] != fold) & (folds['num_images']>min_num_in_train)].index
        val_idx =folds[(folds['fold'] == fold) & (folds['num_images']>min_num_in_train)].index #if no_single_val else folds[folds['singlet_fold'] == fold].index
    elif singles_in_fold:
        trn_idx = folds[(folds['fold'] != fold) & (folds['not_seen']==1)].index
        val_idx =folds[(folds['fold'] == fold) | (folds['not_seen']==0)].index #if no_single_val else folds[folds['singlet_fold'] == fold].index
    else:
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    
    train_dataset = TrainDataset(train_folds,data_root_path, transform=train_transforms,crop_p=crop_p,crop_csv_path=crop_csv_path,crop_backfin_csv_path=crop_backfin_csv_path,hide=min_num_in_train if train_not_seen else 0)
    valid_dataset = TrainDataset(valid_folds, data_root_path,transform=val_transforms,crop_p=1 if use_crop_for_val else 0,crop_csv_path=crop_csv_path,crop_backfin_csv_path=crop_backfin_csv_path,new_targets=singles_in_fold)
    train_dataset_emb = TrainDataset(train_folds,data_root_path, transform=val_transforms,crop_p=1 if use_crop_for_val else 0,crop_csv_path=crop_csv_path,crop_backfin_csv_path=crop_backfin_csv_path)
    
    if use_sampler:
        sampler = StepSampler(data_source=train_dataset,epochs=epochs)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True if not use_sampler else False, 
                              sampler = sampler if use_sampler else None,
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
    
