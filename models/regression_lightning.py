import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import dataset_precip
import argparse
import numpy as np


class UNet_base(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                              mode="min",
                                                              factor=0.1,
                                                              patience=self.hparams['lr_patience']),
            'monitor': 'val_loss',  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch[0].to(self.hparams['device']), batch[1].to(self.hparams['device'])
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output['loss']

        loss_mean /= len(outputs)
        return {"log": {"train_loss": loss_mean},
                "progress_bar": {"train_loss": loss_mean}}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].to(self.hparams['device']), batch[1].to(self.hparams['device'])
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["val_loss"]
        avg_loss /= len(outputs)
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs,
                "progress_bar": {"val_loss": avg_loss}}

    def test_step(self, batch, batch_idx):
        x, y = batch[0].to(self.hparams['device']), batch[1].to(self.hparams['device'])
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        return {"test_loss": val_loss}

    def test_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["test_loss"]
        avg_loss /= len(outputs)
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs,
                "progress_bar": {"test_loss": avg_loss}}


class Precip_regression_base(UNet_base):
    def __init__(self, hparams):
        super(Precip_regression_base, self).__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        if self.hparams['use_oversampled_dataset']:
            self.train_dataset = dataset_precip.precipitation_maps_oversampled_h5(
            folder=self.hparams['dataset_folder'], train=True, in_channels=self.hparams['in_channels']
        )
            self.valid_dataset = dataset_precip.precipitation_maps_oversampled_h5(
            folder=self.hparams['dataset_folder'], train=True, in_channels=self.hparams['in_channels']
        )
        else:
            self.train_dataset = dataset_precip.precipitation_maps_h5(
            folder=self.hparams['dataset_folder'], train=True, in_channels=self.hparams['in_channels'],
            out_channels=self.hparams['out_channels']
        )
            self.valid_dataset = dataset_precip.precipitation_maps_h5(
            folder=self.hparams['dataset_folder'], train=True, in_channels=self.hparams['in_channels'],
            out_channels=self.hparams['out_channels']
        )
        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams['valid_size'] * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams['batch_size'], sampler=self.train_sampler,
            num_workers=4, pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.hparams['batch_size'], sampler=self.valid_sampler,
            num_workers=4, pin_memory=True
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams['batch_size'], sampler=self.test_sampler,
            num_workers=2, pin_memory=True
        )
        return test_loader
