from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import torch
import os
from tqdm import tqdm
from cloud_cover import cloud_cover_dataset
from models import models

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def load_model(model, model_folder, device):
    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    model_file = models[-1]
    model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
    model.eval()
    model.to(torch.device(device))
    return model

def load_data():
    dataset = cloud_cover_dataset.cloud_maps(
        folder=data_file,
        in_channels=4,
        out_channels=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return test_dl

def predict(model, device):
    test_dl = load_data()
    for x, y_true in tqdm(test_dl, leave=False):
        x = x.to(torch.device(device))
        output = model(x)
        fig, axes = plt.subplots(2, 2, figsize=(25, 25))
        axes[0][0].imshow(x[0][0].detach().cpu().numpy())
        axes[0][1].imshow(x[0][1].detach().cpu().numpy())
        axes[1][0].imshow(x[0][2].detach().cpu().numpy())
        axes[1][1].imshow(x[0][3].detach().cpu().numpy())
        #axes[1].imshow(y_true[0][0].cpu().numpy())
        #axes[2].imshow((output[0][0]).detach().cpu().numpy())
        axes[0][0].set_title('1st input image', {'fontsize': 12})
        axes[0][1].set_title('2nd input image', {'fontsize': 12})
        axes[1][0].set_title('3rd input image', {'fontsize': 12})
        axes[1][1].set_title('4th input image', {'fontsize': 12})
        #axes[1].set_title('Ground truth (180 minutes)', {'fontsize': 12})
        #axes[2].set_title('Prediction (180 minutes)', {'fontsize': 12})
        plt.show()

if __name__ == '__main__':
    model = models.SAR_UNet_cloud
    model_folder = "cloud_plot_models"
    data_file = 'data/cloud cover dataset'
    device = 'cpu'
    model = load_model(model,model_folder,device)

    predict(model, device)