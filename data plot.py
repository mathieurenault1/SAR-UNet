import sys
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from utils import dataset_precip
from tqdm import tqdm
from models import models
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger, EarlyStopping
import warnings
import torch.functional as F
import requests
import torchvision

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def load_data():
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        folder=data_file,
        in_channels=12,
        train=False)
    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_dl

def plot():
    test_dl = load_data()
    for x, y_true in tqdm(test_dl, leave=False):
        fig, axes = plt.subplots(2, 1, figsize=(25, 25))
        axes[0].imshow(x[0][0].detach().cpu().numpy())
        axes[0].set_title('1st input image', {'fontsize': 12})
        plt.show()

if __name__ == '__main__':
    data_file = 'data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5'
    plot()