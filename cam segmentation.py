from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from utils import dataset_precip
from tqdm import tqdm
from models import models
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

def get_segmentation_data(in_channels):
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        folder=data_file,
        in_channels=in_channels,
        train=False)

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
        mask = np.digitize((output[0][0]*47.83*12).detach().cpu().numpy(), np.array([1.5]), right=True)
        mask_float = np.float32(mask)

        image = torch.stack([x[0][0], x[0][0], x[0][0]],dim=2)
        image = image.cpu().numpy()

        #model.up4.conv.double_conv[3].pointwise
        targets = [SemanticSegmentationTarget(0, mask_float, device)]
        use_cuda = (device == 'cuda')
        cam_image = []
        for layer in target_layers:
            with GradCAM(model=model,target_layers=layer,use_cuda=use_cuda) as cam:
                grayscale_cam = cam(input_tensor=x,targets=targets)[0, :]
                cam_image.append(show_cam_on_image(image, grayscale_cam, use_rgb=True))

        fig, axes = plt.subplots(2, 3, figsize=(25, 25))
        # axes[0].imshow(x[0][0].detach().cpu().numpy())
        # axes[0].imshow(y_true[0].cpu().numpy())
        # axes[1].imshow((output[0][0]).detach().cpu().numpy())
        # axes[0].set_title('Ground Truth', {'fontsize': 16})
        # axes[0][2].imshow(mask_float)
        # axes[1].set_title('Prediction', {'fontsize': 16})

        axes[0][0].imshow(cam_image[6])
        #axes[0][0].set_title('Residual DSC Blocks Activations', {'fontsize': 12})
        axes[0][1].imshow(cam_image[7])
        #axes[0][1].set_title('DSC Path Activation', {'fontsize': 12})
        axes[0][2].imshow(cam_image[8])
        #axes[0][2].set_title('Residual Connection Activation', {'fontsize': 12})
        # axes[0][3].imshow(cam_image[3])
        # axes[0][3].set_title('CBAM Activation', {'fontsize': 12})
        axes[1][0].imshow(cam_image[9])
        axes[1][1].imshow(cam_image[10])
        axes[1][2].imshow(cam_image[11])
        # axes[1][3].imshow(cam_image[7])
        # axes[0][0].imshow(cam_image[8])
        # axes[0][1].imshow(cam_image[9])
        # axes[0][2].imshow(cam_image[10])
        # axes[0][3].imshow(cam_image[11])
        # axes[1][0].imshow(cam_image[12])
        # axes[1][1].imshow(cam_image[13])
        # axes[1][2].imshow(cam_image[14])
        # axes[1][3].imshow(cam_image[15])
        # axes[4][0].imshow(cam_image[16])
        # axes[4][1].imshow(cam_image[17])
        # axes[4][2].imshow(cam_image[18])
        # axes[4][3].imshow(cam_image[19])

        # axes[5][0].imshow(cam_image[20])
        # axes[5][0].set_title('Residual DSC Block 6 Activation', {'fontsize': 12})
        # axes[6][0].imshow(cam_image[23])
        # axes[6][0].set_title('Residual DSC Block 7 Activation', {'fontsize': 12})
        # axes[7][0].imshow(cam_image[26])
        # axes[7][0].set_title('Residual DSC Block 8 Activation', {'fontsize': 12})
        # axes[8][0].imshow(cam_image[29])
        # axes[8][0].set_title('Residual DSC Block 9 Activation', {'fontsize': 12})
        # axes[0][2].set_title('binarized output', {'fontsize': 16})

        # axes[1][1].imshow(x[0][11].detach().cpu().numpy())
        # axes[1][1].set_title('12th input image', {'fontsize': 16})
        # axes[1][2].imshow(cam_image[3])
        # axes[1][2].set_title('GradCAM', {'fontsize': 16})
        plt.show()



if __name__ == '__main__':
    model = models.SAR_UNet_precip
    in_channels = 12
    model_folder = "model_cam"
    data_file = 'data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5'
    device = 'cpu'
    model = load_model(model,model_folder,device)
    print(model)
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
