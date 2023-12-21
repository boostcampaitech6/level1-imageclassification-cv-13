"""
This script is for defining some constants and functions that are used in other scripts.
"""
import os
import random
from enum import Enum

import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MaskLabels(int, Enum):
    wear: int = 0
    incorrect: int = 1
    not_wear: int = 2

    @classmethod
    def get_label(self, label: str) -> int:
        if label.startswith("mask"):
            return self.wear
        elif label.startswith('incorrect'):
            return self.incorrect
        elif label.startswith('normal'):
            return self.not_wear
        else:
            raise KeyError(f'{label} is not exist in MaskLabels')


class GenderLabel(int, Enum):
    male: int = 0
    female: int = 1

    @classmethod
    def get_label(self, gender: str) -> int:
        if gender == "male":
            return self.male
        else:
            return self.female


class AgeLabel(int, Enum):
    young: int = 0
    middle: int = 1
    old: int = 2

    @classmethod
    def get_label(self, age: int) -> int:
        if age < 30:
            return self.young
        elif age < 60:
            return self.middle
        else:
            return self.old


def extract_data(root_dir, candidates: list):
    ''' This Function removes the train data that starts with a dot('._xxx')
    Args:
        root_dir (os.PathLike): root directory of train data
        candidates (list) : list of candidates
    Returns:
        img_paths (list): list of [label, image_path]
    '''
    img_paths = []
    for candidate in candidates:
        for sub_candidate in os.listdir(os.path.join(root_dir, candidate)):
            if not sub_candidate.startswith('.'):
                img_paths.append([candidate, os.path.join(root_dir, candidate, sub_candidate)])
    print("Extracting Finish. Length of train_data: ", len(img_paths))
    return img_paths


def extract_eval_data(eval_csv):
    ''' This Function extracts the eval data from eval_csv
    Args:
        eval_csv (os.PathLike): eval csv file path
    Returns:
        img_paths (list): list of [label, image_path]
    '''
    df = pd.read_csv(eval_csv)
    return [os.path.join("datasets", "eval", "images", x) for x in df['ImageID']]


def create_concat_df(raw_df_path, concat_df_path, val_df_path, root_dir, candidates, config):
    ''' This Function creates the dataframe for train/valid data
    Args:
        raw_df_path (os.PathLike): raw dataframe path
        concat_df_path (os.PathLike): concat dataframe path (train)
        val_df_path (os.PathLike): concat dataframe path (valid)
        root_dir (os.PathLike): root directory of train data
        candidates (list): list of candidates
    Returns:
        None
    '''
    # raw_df = pd.read_csv(raw_df_path)
    # train, valid = train_test_split(raw_df, test_size=0.2, stratify=raw_df[] random_state=config['seed'])
    # for method, df in zip(['train', 'valid'], [train, valid]):
    #     total_data = extract_data(root_dir, candidates)
    #     df_rows = []

    #     for folder_name in df["path"]:
    #         candits = list(filter(lambda x: x[0]==folder_name, total_data))
    #         _, gender, _, age = folder_name.split("_")
    #         gender = GenderLabel.get_label(gender)
    #         age = AgeLabel.get_label(int(age))
    #         for _, abs_path in candits:
    #             mask = abs_path.split('/')[-1].split('.')[0]
    #             mask = MaskLabels.get_label(mask)
    #             total_label = mask*6 + gender*3 + age
    #             df_rows.append([abs_path, gender, age, mask, total_label])

    #     new_df = pd.DataFrame(df_rows, columns=['path', 'gender_label', 'age_label', 'mask_label','label'])
    #     if method == "train":
    #         new_df.to_csv(concat_df_path, index=False)
    #     else:
    #         new_df.to_csv(val_df_path, index=False)
    raw_df = pd.read_csv(raw_df_path)
    total_data = extract_data(root_dir, candidates)
    df_rows = []
    
    for folder_name in raw_df["path"]:
        candits = list(filter(lambda x: x[0]==folder_name, total_data))
        _, gender, _, age = folder_name.split("_")
        gender = GenderLabel.get_label(gender)
        age = AgeLabel.get_label(int(age))
        for _, abs_path in candits:
            mask = abs_path.split('/')[-1].split('.')[0]
            mask = MaskLabels.get_label(mask)
            total_label = mask*6 + gender*3 + age
            df_rows.append([abs_path, gender, age, mask, total_label])

    new_df = pd.DataFrame(df_rows, columns=['path', 'gender_label', 'age_label', 'mask_label','label'])
    new_df.to_csv(concat_df_path, index=False)
    
    print("Successfully created new train dataframe")


def create_sub_df(concat_df_path, col_name, config):
    ''' This Function creates the dataframe for split train/valid data with specific column.
    Args:
        concat_df_path (os.PathLike): raw dataframe path
        col_name (str): column name
        config (dict): train config
    Returns:
        None
    '''
    concat_df = pd.read_csv(concat_df_path)
    train_x, valid_x, train_y, valid_y = train_test_split(
        concat_df['path'],
        concat_df[col_name],
        stratify=concat_df[col_name],
        test_size=0.2,
        random_state=config['seed']
    )

    train_df = pd.DataFrame({'path': train_x, col_name: train_y})
    valid_df = pd.DataFrame({'path': valid_x, col_name: valid_y})

    train_df.to_csv(os.path.join("datasets", "train", f"new_train_{col_name}.csv"), index=False)
    valid_df.to_csv(os.path.join("datasets", "train", f"new_valid_{col_name}.csv"), index=False)


def split_data(raw_df_path, concat_df_path, val_df_path, root_dir, config):
    ''' This Function splits the train data into train/valid data
    Args:
        raw_df_path (os.PathLike): raw dataframe path
        concat_df_path (os.PathLike): concat dataframe path (train)
        root_dir (os.PathLike): root directory of train data
    Returns:
        candidates (list): list of candidates
        img_paths (list): list of [label, image_path]
    '''
    candidates = list(filter(lambda x: not x.startswith('.'), os.listdir(root_dir)))
    create_concat_df(raw_df_path, concat_df_path, val_df_path, root_dir, candidates, config)
    # for col_name in ['gender_label', 'age_label', 'mask_label']:
    #     create_sub_df(concat_df_path, col_name, config)


def set_seed(seed: int):
    """Set seed for reproducibility.
    Args:
        seed (int): seed number
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_train_csv(config):
    ''' Check train/valid csv file./
    Args:
        config (dict): config dictionary
    '''
    train_root = config["train_root"]
    try:
        assert os.path.isfile(os.path.join(train_root, "new_train.csv"))
    except:
        split_data(
            os.path.join(config["train_root"], config["raw_df"]),
            os.path.join(config["train_root"], config["concat_df"]),
            os.path.join(config["train_root"], config["valid_df"]),
            os.path.join(config["train_root"], config["train_img_dir"]),
            config
        )

        

def get_config(config_path: os.PathLike) -> dict:
    """Get config from config file.
    Args:
        config_path (os.PathLike): config file path
    Returns:
        config (dict): config dict
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    config = get_config("configs/default.yaml")
    split_data(
        os.path.join(config["train_root"], config["raw_df"]),
        os.path.join(config["train_root"], config["concat_df"]),
        os.path.join(config["train_root"], config["valid_df"]),
        os.path.join(config["train_root"], config["train_img_dir"]),
        config
    )
