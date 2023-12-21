import wandb
import torch

from .dataset import cutmix
from .loss import _criterion_entrypoints
from .model import BaseModel
from lightning import LightningModule
from lion_pytorch import Lion
from torcheval.metrics.functional import multiclass_f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskPLModule(LightningModule):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()
        self.model = BaseModel(config['multi_class'])
        self.config = config
        self.gender_criterion = _criterion_entrypoints[self.config['g_loss']](classes=self.config['g_class'])
        self.mask_criterion = _criterion_entrypoints[self.config['m_loss']](classes=self.config['m_class'])
        self.age_criterion = _criterion_entrypoints[self.config['a_loss']](classes=self.config['a_class'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # images, gender, age, mask = cutmix(batch, alpha=1.0)
        # pred = self.model(images)
        # (pred_gender, pred_age, pred_mask) = torch.split(pred, [2, 3, 3], dim=1)
        
        # ## mask loss
        # mlabel, shuffled_mlabel, lam = mask
        # m_loss = lam * self.mask_criterion(pred_mask, mlabel) + (1 - lam) * self.mask_criterion(pred_mask, shuffled_mlabel)
                
        # ## gender loss
        # glabel, shuffled_glabel, lam = gender
        # g_loss = lam * self.gender_criterion(pred_gender, glabel) + (1 - lam) * self.gender_criterion(pred_gender, shuffled_glabel)
        
        # ## age loss
        # alabel, shuffled_alabel, lam = age
        # a_loss = lam * self.age_criterion(pred_age, alabel) + (1 - lam) * self.age_criterion(pred_age, shuffled_alabel)
        
        # m_acc = multiclass_f1_score(pred_mask.argmax(1), mask[0], num_classes=self.config['m_class'])
        # g_acc = multiclass_f1_score(pred_gender.argmax(1), gender[0], num_classes=self.config['g_class'])
        # a_acc = multiclass_f1_score(pred_age.argmax(1), age[0], num_classes=self.config['a_class'])
        
        images, *labels = batch
        gender, age, mask = labels
        pred = self.model(images)
        (pred_gender, pred_age, pred_mask) = torch.split(pred, [2, 3, 3], dim=1)
        
        m_loss = self.mask_criterion(pred_mask, mask)
        g_loss = self.gender_criterion(pred_gender, gender)
        a_loss = self.age_criterion(pred_age, age)
        
        m_acc = multiclass_f1_score(pred_mask.argmax(1), mask, num_classes=self.config['m_class'])
        g_acc = multiclass_f1_score(pred_gender.argmax(1), gender, num_classes=self.config['g_class'])
        a_acc = multiclass_f1_score(pred_age.argmax(1), age, num_classes=self.config['a_class'])

        avg_loss = (m_loss*1.33 + g_loss + a_loss*1.66) / 3
        avg_acc = (m_acc + g_acc + a_acc) / 3
        self.log("t_m_loss", m_loss, on_step=True, on_epoch=True, logger=True)
        self.log("t_g_loss", g_loss, on_step=True, on_epoch=True, logger=True)
        self.log("t_a_loss", a_loss, on_step=True, on_epoch=True, logger=True)
        self.log("t_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("t_acc", avg_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
            gender = gender.detach().cpu().numpy()
            age = age.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            pred_gender = pred_gender.argmax(1).detach().cpu().numpy()
            pred_age = pred_age.argmax(1).detach().cpu().numpy()
            pred_mask = pred_mask.argmax(1).detach().cpu().numpy()
            values = [pp*6 + qq*3 + rr for pp, qq, rr in zip(mask, gender, age)]
            pred_values = [pp*6 + qq*3 + rr for pp, qq, rr in zip(pred_mask, pred_gender, pred_age)]

            data = [[wandb.Image(images[j]), values[j], pred_values[j]] for j in range(8)]
            self.logger.log_table(key='train', columns=['image', 'label', 'pred'], data=data)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        images, *label = batch
        gender, age, mask = label
        pred = self.model(images)
        (pred_gender, pred_age, pred_mask) = torch.split(pred, [2, 3, 3], dim=1)

        m_loss = self.mask_criterion(pred_mask, mask)
        g_loss = self.gender_criterion(pred_gender, gender)
        a_loss = self.age_criterion(pred_age, age)

        m_acc = multiclass_f1_score(pred_mask.argmax(1), mask, num_classes=self.config['m_class'])
        g_acc = multiclass_f1_score(pred_gender.argmax(1), gender, num_classes=self.config['g_class'])
        a_acc = multiclass_f1_score(pred_age.argmax(1), age, num_classes=self.config['a_class'])

        avg_loss = (m_loss*1.5 + g_loss + a_loss*2) / 3
        avg_acc = (m_acc + g_acc + a_acc) / 3
        self.log("v_m_loss", m_loss, on_step=True, on_epoch=True, logger=True)
        self.log("v_g_loss", g_loss, on_step=True, on_epoch=True, logger=True)
        self.log("v_a_loss", a_loss, on_step=True, on_epoch=True, logger=True)
        self.log("v_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("v_acc", avg_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return avg_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr_rate'])
        optimizer = Lion(self.model.parameters(), lr=self.config['lr_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "scheduler": scheduler}
