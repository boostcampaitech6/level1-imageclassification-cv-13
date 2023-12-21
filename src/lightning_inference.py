
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from .utils import get_config, set_seed, extract_eval_data
from .dataset import MaskInferenceDataset, get_valid_transform
from .lightning_train import MaskPLModule


def pl_inference():
    config = get_config("configs/multiclass.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    preds = []
    model = MaskPLModule.load_from_checkpoint('ckpt/ViT_pl_epoch=89_t_loss=0.070969_t_acc=0.977511_v_loss=0.099129_v_acc=0.973509.ckpt', config=config)
    model.to(device)

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
        for idx, images in tqdm(enumerate(inference_dataloader)):
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
