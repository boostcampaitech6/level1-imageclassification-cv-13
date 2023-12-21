
import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import BaseModel
from .utils import get_config, set_seed, extract_eval_data
from .dataset import MaskInferenceDataset, get_valid_transform


def inference():
    config = get_config("configs/multiclass.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    preds = []

    model = BaseModel(config['multi_class']).to(device)
    model.load_state_dict(torch.load('ckpt/ViT_epoch32_tloss:0.021_tacc0.971_vloss0.025_vacc0.971.pth'))
    model.eval()

    inference_dataset = MaskInferenceDataset(
        extract_eval_data(config['eval_df']),
        get_valid_transform()
    )
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    with torch.no_grad():
        for images in inference_dataloader:
            images = images.to(device)
            pred = model(images)
            (pred_gender, pred_age, pred_mask) = torch.split(pred, [2, 3, 3], dim=1)

            age_pred = pred_age.argmax(1).detach().cpu().numpy()
            gender_pred = pred_gender.argmax(1).detach().cpu().numpy()
            mask_pred = pred_mask.argmax(1).detach().cpu().numpy()
            preds.extend([x*6+y*3+z for x, y, z in zip(mask_pred, gender_pred, age_pred)])

    info_df = pd.read_csv(config['eval_df'])
    info_df['ans'] = preds
    info_df.to_csv(config['save_df'], index=False)
