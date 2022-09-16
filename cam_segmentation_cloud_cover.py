from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from tqdm import tqdm
from models import models
from cloud_cover import cloud_cover_dataset
import matplotlib.pyplot as plt
import warnings
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class SemanticSegmentationTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if device == 'cuda':
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def load_model(model, model_folder, device):
    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    model_file = models[-1]
    model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
    model.eval()
    model.to(torch.device(device))
    return model

def get_segmentation_data():
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

def run_cam(model, target_layers, device):
    test_dl = get_segmentation_data()
    for x, y_true in tqdm(test_dl, leave=False):
        x = x.to(torch.device(device))
        output = model(x)
        mask = np.digitize((output[0][0]).detach().cpu().numpy(), np.array([1]), right=True)
        mask_float = np.float32(mask)

        image = torch.stack([x[0][0], x[0][0], x[0][0]],dim=2)
        image = image.cpu().numpy()

        targets = [SemanticSegmentationTarget(0, mask_float, device)]
        use_cuda = (device == 'cuda')
        cam_image = []
        for layer in target_layers:
            with GradCAM(model=model,target_layers=layer,use_cuda=use_cuda) as cam:
                grayscale_cam = cam(input_tensor=x,targets=targets)[0, :]
                cam_image.append(show_cam_on_image(image, grayscale_cam, use_rgb=True))

        fig, axes = plt.subplots(4, 3, figsize=(25, 25))
        axes[0][0].imshow(cam_image[0])
        axes[0][1].imshow(cam_image[1])
        axes[0][2].imshow(cam_image[2])
        axes[1][0].imshow(cam_image[3])
        axes[1][1].imshow(cam_image[4])
        axes[1][2].imshow(cam_image[5])
        axes[2][0].imshow(cam_image[6])
        axes[2][1].imshow(cam_image[7])
        axes[2][2].imshow(cam_image[8])
        axes[3][0].imshow(cam_image[9])
        axes[3][1].imshow(cam_image[10])
        axes[3][2].imshow(cam_image[11])
        #axes[0][0].imshow(cam_image[0])
        #axes[0][1].imshow(cam_image[1])
        #axes[1][0].imshow(cam_image[2])
        #axes[1][1].imshow(cam_image[3])
        plt.show()

if __name__ == '__main__':
    hparams = {'model': 'SAR_UNet_cloud',
               'out_channels': 6,
               'in_channels': 4,
               "batch_size": 6,
               "learning_rate": 0.001,
               'gpus': -1,
               "lr_patience": 4,
               "es_patience": 30,
               "use_oversampled_dataset": True,
               "bilinear": True,
               "valid_size": 0.1,
               "dataset_folder": "data/cloud cover dataset",
               "resume_from_checkpoint": None
               }
    model = models.SAR_UNet_cloud(hparams=hparams)
    model_folder = "model_cam/cloud_cover"
    data_file = 'data/cloud cover dataset'
    device = 'cpu'
    model = load_model(model, model_folder, device)
    print(model)
    #target_layers = [[model.RRCNN1], [model.RRCNN2], [model.RRCNN5], [model.Up_RRCNN4]]
    target_layers = [#[model.RRCNN1],[model.RRCNN1.doubleconv],[model.RRCNN1.Conv_1x1],[model.cbam1],
                     # [model.RRCNN2],[model.RRCNN2.doubleconv],[model.RRCNN2.Conv_1x1],[model.cbam2],
                     # [model.RRCNN3],[model.RRCNN3.doubleconv],[model.RRCNN3.Conv_1x1],[model.cbam3],
                     # [model.RRCNN4],[model.RRCNN4.doubleconv],[model.RRCNN4.Conv_1x1],[model.cbam4],
                     # [model.RRCNN5],[model.RRCNN5.doubleconv],[model.RRCNN5.Conv_1x1],[model.cbam5],
                     [model.Up_RRCNN5],[model.Up_RRCNN5.doubleconv],[model.Up_RRCNN5.Conv_1x1],
                     [model.Up_RRCNN4],[model.Up_RRCNN4.doubleconv],[model.Up_RRCNN4.Conv_1x1],
                     [model.Up_RRCNN3],[model.Up_RRCNN3.doubleconv],[model.Up_RRCNN3.Conv_1x1],
                     [model.Up_RRCNN2],[model.Up_RRCNN2.doubleconv],[model.Up_RRCNN2.Conv_1x1],
                     ]
    run_cam(model, target_layers, device)
