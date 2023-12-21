import os
import argparse

import lightning as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.lightning_inference import pl_inference
from src.ensemble import ensemble
from src.dataset import MaskMultiLabelPLDataset, get_train_transform, get_valid_transform
from src.utils import get_config, set_seed, check_train_csv
from src.lightning_train import MaskPLModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', required=True, help='option')
    args = parser.parse_args()
    
    if args.o == 'pl_train':
        config = get_config("configs/multiclass.yaml")
        set_seed(config['seed'])
        check_train_csv(config)

        dataset = MaskMultiLabelPLDataset(
            config,
            get_train_transform(),
            get_valid_transform()
        )
        dataset.prepare_data()
        dataset.setup()

        train_config = {
            'seed': config['seed'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'lr_rate': config['lr_rate'],
            'g_loss': config['g_loss'],
            'm_loss': config['m_loss'],
            'a_loss': config['a_loss'],
            'optimizer': config['optimizer'],
            'model': "resnet_multiclass",
        }

        checkpoint_callback = ModelCheckpoint(
            dirpath="ckpt/",
            save_top_k=3,
            monitor="v_loss",
            mode='min',
            filename=f"{config['model']}"+"_pl_{epoch}_{t_loss:3f}_{t_acc:2f}_{v_loss:3f}_{v_acc:3f}"
        )
        wandb_logger = WandbLogger(
            project="mask-classification",
            entity="zeroone",
            name=f"{config['model']}_multiclass",
            notes=f"{config['model']}_multiclass with sample data augmentation",
            config=train_config
        )

        trainer = pl.Trainer(
            strategy='auto',
            accelerator="gpu",
            devices=1,
            max_epochs=config['epochs'],
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            precision='16-mixed'
        )
        pl_model = MaskPLModule(config)
        trainer.fit(pl_model, datamodule=dataset)

    elif args.o == 'pl_inference':
        pl_inference()
        
    elif args.o == 'ensemble':
        ensemble()
