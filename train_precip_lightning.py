from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger, EarlyStopping
from pytorch_lightning import loggers
import torch
from models import models
import pytorch_lightning as pl
import torchsummary
import os


def train_regression(hparams):
    if hparams['model'] == "SAR_UNet_precip":
        net = models.SAR_UNet_precip(hparams=hparams)
    elif hparams['model'] == "SmaAt_UNet_precip":
        net = models.SmaAt_UNet_precip(hparams=hparams)
    elif hparams['model'] == "UNet_precip":
        net = models.UNet_precip(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams['model']}' not implemented")

    net = net.to(device)
    torchsummary.summary(net, (hparams['in_channels'], 288, 288), device="cpu")

    default_root_dir = ""

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd() + "/" + default_root_dir + "/" + net.__class__.__name__ + "/{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=net.__class__.__name__ + "_rain_threshold_50_"
    )
    lr_logger = LearningRateLogger()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_root_dir, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=hparams['es_patience'],
                                           # is effectively half (due to a bug in pytorch-lightning)
                                           )
    trainer = pl.Trainer(gpus=(-1 if device=='cuda:0' else 0),
                         weights_summary=None,
                         max_epochs=200,
                         default_root_dir=default_root_dir,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=earlystopping_callback,
                         logger=tb_logger,
                         callbacks=[lr_logger],
                         resume_from_checkpoint=hparams['resume_from_checkpoint']
                         )
    trainer.fit(net)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hparams = {
        'device': device,
        'model': 'SAR_UNet_precip',
        'out_channels': 1,
        'in_channels': 12,  # or 6 or 18 for more or less input data
        "batch_size": 6,
        "learning_rate": 0.001,
        'gpus': (-1 if device=='cuda:0' else 0),
        "lr_patience": 4,
        "es_patience": 30,
        "use_oversampled_dataset": True,
        "bilinear": True,
        "valid_size": 0.1,
        "dataset_folder": "data/precip/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5",
        # change input-length and img-ahead accordingly
        "resume_from_checkpoint": None
        # f"{args.model}/ResSmaAt_UNet2_rain_threshold_50_epoch=56-val_loss=0.300085.ckpt"
    }
    train_regression(hparams=hparams)
