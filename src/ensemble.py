import math
import pandas as pd
from collections import Counter

from .utils import get_config


def encoding(value: int) -> list:
    mask, gender, age = 0, 0, 0
    mask = value // 6
    value %= 6
    gender = value // 3
    value %= 3
    age = value
    
    return mask, gender, age

def ensemble():    
    config = get_config("configs/multiclass.yaml")
    
    df1 = pd.read_csv('datasets/ensemble/vit_age_straigfy.csv')
    df2 = pd.read_csv('datasets/ensemble/vit_mask_straify.csv')
    df3 = pd.read_csv('datasets/ensemble/vit_mask_straify_cutmix.csv')
    df4 = pd.read_csv('datasets/ensemble/yboutput.csv')
    df5 = pd.read_csv('datasets/ensemble/dhoutput.csv')
    
    # soft voting
    preds = []
    for r1, r2, r3, r4, r5 in zip(df1['ans'], df2['ans'], df3['ans'], df4['ans'], df5['ans']):
        r1, r2, r3, r4, r5 = encoding(r1), encoding(r2), encoding(r3), encoding(r4), encoding(r5)
        mask_pred = Counter([r1[0], r2[0], r3[0], r4[0], r5[0]]).most_common(1)[0][0]
        gender_pred = Counter([r1[1], r2[1], r3[1], r4[1], r5[1]]).most_common(1)[0][0]
        # age_pred = Counter([r1[2], r2[2], r3[2], r4[2], r5[2]]).most_common(1)[0][0]
        age_pred = math.ceil(sum([r1[2], r2[2], r3[2], r4[2], r5[2]])/5)
             
        preds.append(mask_pred*6 + gender_pred*3 + age_pred)

    info_df = pd.read_csv(config['eval_df'])
    info_df['ans'] = preds
    info_df.to_csv(config['save_df'], index=False)
