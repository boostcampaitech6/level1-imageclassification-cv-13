import os
import sys

import wandb
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .loss import _criterion_entrypoints
from .model import BaseModel
from .utils import set_seed, get_config, check_train_csv
from .dataset import MaskMultiLabelDataset, get_train_transform, get_valid_transform


def set_wandb(config: dict):
    ''' Set wandb config.
    '''
    train_config = {
        'seed': config['seed'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'lr_rate': config['lr_rate'],
        'g_loss': config['gender_loss'],
        'm_loss': config['mask_loss'],
        'a_loss': config['age_loss'],
        'optimizer': config['optimizer'],
        'model': "resnet_multiclass",
    }

    wandb.init(
        project="mask-classification",
        entity="zeroone",
        name="resnet_multiclass",
        notes="resnet_multiclass with sample data augmentation",
        config=train_config
    )


def train():
    config = get_config("configs/multiclass.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    try:
        set_wandb(config)
    except Exception as e:
        print("wandb init error")
        sys.exit(0)

    set_seed(config["seed"])
    check_train_csv(config)

    train_df = pd.read_csv(os.path.join(config["train_root"], "new_train.csv"))
    img_paths, target = train_df['path'], train_df[['gender_label', 'age_label', 'mask_label']]
    seeds = [config['seed'] + i for i in range(config['k_fold'])]

    for fold_num in range(config['k_fold']):
        x_train, x_valid, y_train, y_valid = train_test_split(
            img_paths, target,
            test_size=0.2,
            random_state=seeds[fold_num],
            stratify=train_df['age_label']
        )

        train_data = pd.concat([x_train, y_train], axis=1)
        valid_data = pd.concat([x_valid, y_valid], axis=1)

        train_dataset = MaskMultiLabelDataset(train_data, get_train_transform())
        valid_dataset = MaskMultiLabelDataset(valid_data, get_valid_transform())

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=True
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=True
        )

        model = BaseModel(config["multi_class"]).to(device)
        gender_criterion = _criterion_entrypoints[config['gender_loss']]()
        mask_criterion = _criterion_entrypoints[config['mask_loss']]()
        age_criterion = _criterion_entrypoints[config['age_loss']]()
        epoch = config["epochs"]
        optim = torch.optim.Adam(model.parameters(), lr=config["lr_rate"])

        for i in range(epoch):
            train_loss, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            for idx, batch in tqdm(enumerate(train_dataloader)):
                images, *label = batch
                gender, age, mask = label
                images, gender, age, mask = images.to(device), gender.to(device), age.to(device), mask.to(device)
                optim.zero_grad()
                pred = model(images)
                (pred_gender, pred_age, pred_mask) = torch.split(pred, [2, 3, 3], dim=1)

                mask_loss = mask_criterion(pred_mask, mask)
                gender_loss = gender_criterion(pred_gender, gender)
                age_loss = age_criterion(pred_age, age)

                mask_correct = (pred_mask.argmax(1) == mask).sum().item()
                gender_correct = (pred_gender.argmax(1) == gender).sum().item()
                age_correct = (pred_age.argmax(1) == age).sum().item()

                total_loss = (mask_loss + gender_loss + age_loss*1.5) / 3
                total_loss.backward()
                optim.step()

                total_correct = (mask_correct + gender_correct + age_correct) / (3 * config['batch_size'])
                train_loss += total_loss.item()
                train_acc += total_correct

                ## wandb image logging
                if idx % 20 == 0:
                    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()

                    gender = gender.detach().cpu().numpy()
                    age = age.detach().cpu().numpy()
                    mask = mask.detach().cpu().numpy()

                    pred_gender = pred_gender.argmax(1).detach().cpu().numpy()
                    pred_age = pred_age.argmax(1).detach().cpu().numpy()
                    pred_mask = pred_mask.argmax(1).detach().cpu().numpy()
                    values = [pp*6 + qq*3 + rr for pp, qq, rr in zip(mask, gender, age)]
                    pred_values = [pp*6 + qq*3 + rr for pp, qq, rr in zip(pred_mask, pred_gender, pred_age)]

                    tbl = wandb.Table(columns=["image", "label", "pred"])
                    for j in range(4):
                        tbl.add_data(wandb.Image(images[j]), values[j], pred_values[j])
                    wandb.log({f"output": tbl})

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            with torch.no_grad():
                for images, *label in tqdm(valid_dataloader):

                    gender, age, mask = label
                    images, gender, age, mask = images.to(device), gender.to(device), age.to(device), mask.to(device)
                    pred = model(images)
                    (pred_gender, pred_age, pred_mask) = torch.split(pred, [2, 3, 3], dim=1)

                    mask_loss = mask_criterion(pred_mask, mask)
                    gender_loss = gender_criterion(pred_gender, gender)
                    age_loss = mask_criterion(pred_age, age)

                    mask_correct = (pred_mask.argmax(1) == mask).sum().item()
                    gender_correct = (pred_gender.argmax(1) == gender).sum().item()
                    age_correct = (pred_age.argmax(1) == age).sum().item()

                    total_loss = (mask_loss + gender_loss + age_loss*1.5) / 3
                    total_correct = (mask_correct + gender_correct + age_correct) / (3 * config['batch_size'])
                    val_loss += total_loss.item()
                    val_acc += total_correct

                val_loss /= len(valid_dataloader)
                val_acc /= len(valid_dataloader)

            torch.save(
                model.state_dict(),
                f"ckpt/{config['model']}_epoch{i}_tloss:{train_loss:.3f}_tacc{train_acc:.3f}_vloss{val_loss:.3f}_vacc{val_acc:.3f}.pth"
            )

            ## wandb logging
            wandb.log({f"train_loss": train_loss})
            wandb.log({f"train_acc": train_acc})
            wandb.log({f"val_loss": val_loss})
            wandb.log({f"val_acc": val_acc})

        wandb.finish()


if __name__ == '__main__':
    train()
