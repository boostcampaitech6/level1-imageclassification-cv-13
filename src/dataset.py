"""
This script is for the dataset class.
"""
import os
from typing import Union

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


class MaskDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Union[transforms.Compose, A.Compose]=None
    ) -> None:
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.df.iloc[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)
        return image, label


class MaskMultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        img_path, *label = self.df.iloc[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)["image"]
        return image, *label



class MaskInferenceDataset(Dataset):
    def __init__(self, img_paths: list, transform=None) -> None:
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.img_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)["image"]
        return image


class MaskMultiLabelPLDataset(LightningDataModule):
    def __init__(
        self,
        config: dict,
        train_transform=None,
        valid_transform=None
    ) -> None:
        super().__init__()
        self.train_df = pd.read_csv(os.path.join(config["train_root"], config['concat_df']))
        # self.valid_df = pd.read_csv(os.path.join(config["train_root"], config['valid_df']))
        self.config = config
        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        img_paths, target = self.train_df['path'], self.train_df[['gender_label', 'age_label', 'mask_label']]
        x_train, x_valid, y_train, y_valid = train_test_split(
            img_paths, target,
            test_size=0.2,
            random_state=self.config['seed'],
            stratify=self.train_df['age_label']
        )

        train_data = pd.concat([x_train, y_train], axis=1)
        valid_data = pd.concat([x_valid, y_valid], axis=1)   
        self.train_dataset = MaskMultiLabelDataset(train_data, self.train_transform)
        self.valid_dataset = MaskMultiLabelDataset(valid_data, self.valid_transform)
        
        # self.train_dataset = MaskMultiLabelDataset(self.train_df, self.train_transform)
        # self.valid_dataset = MaskMultiLabelDataset(self.valid_df, self.valid_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            drop_last=True
        )


def get_train_transform() -> A.Compose:
    transform = A.Compose([
        A.Resize(384, 384),
        A.CenterCrop(336, 336),
        A.CLAHE(p=0.3),
        A.GaussNoise(p=0.3),
        A.Cutout(num_holes=15, max_h_size=15, max_w_size=15, p=0.3),
        A.Normalize(
            mean=(0.56019358, 0.52410121, 0.501457),
            std=(0.23318603, 0.24300033, 0.24567522)
        ),
        ToTensorV2()
    ])
    return transform


def get_valid_transform() -> A.Compose:
    transform = A.Compose([
        A.Resize(384, 384),
        A.CenterCrop(336, 336),
        A.Normalize(
            mean=(0.56019358, 0.52410121, 0.501457),
            std=(0.23318603, 0.24300033, 0.24567522)
        ),
        ToTensorV2(),
    ])
    return transform


def cutmix(batch, alpha=1.0):
    images, *labels = batch
    gender, age, mask, _ = labels

    # 이미지의 크기 및 랜덤 비율 설정
    B, C, H, W = images.shape
    lam = np.random.beta(alpha, alpha)
    
    # 랜덤 위치 설정
    cx = np.random.uniform(0, W)
    cy = np.random.uniform(0, H)
    w = W * np.sqrt(1 - lam)
    h = H * np.sqrt(1 - lam)
    x0 = int(np.clip(cx - w / 2, 0, W))
    x1 = int(np.clip(cx + w / 2, 0, W))
    y0 = int(np.clip(cy - h / 2, 0, H))
    y1 = int(np.clip(cy + h / 2, 0, H))

    # 이미지 샘플링 및 혼합
    indices = torch.randperm(B)
    mixed_images = images.clone()
    mixed_images[:, :, y0:y1, x0:x1] = images[indices, :, y0:y1, x0:x1]

    # 레이블 조정
    lam = 1 - ((x1 - x0) * (y1 - y0) / (W * H))
    mixed_gender = (gender, gender[indices], lam)
    mixed_age = (age, age[indices], lam)
    mixed_mask = (mask, mask[indices], lam)
        
    return mixed_images, mixed_gender, mixed_age, mixed_mask
