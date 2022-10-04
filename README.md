# SAR-UNet
Code for the Paper "SAR-UNet: Small Attention Residual UNet for Explainable Precipitation Nowcasting"

![sar unet](https://user-images.githubusercontent.com/73837432/193812363-4e9a817d-fd9e-47de-a621-e32766258d0a.png)

# Datasets & Models
If you want access to the datasets (precipitation maps & cloud cover dataset) used in this paper, please visit https://github.com/HansBambel/SmaAt-UNet for further details.
Please put the dataset into "\data\precip" directory for training and testing the precipitation network, and put the trained models into "\checkpoints\eval\precip" directory for future loading.

# Training
For training on the precipitation task we used the train_precip_lightning.py file.
For training on the cloud cover task we used the train_cloud_cover.py file.
Training was done using Pytorch-Lightning. All details about parameters used and device configurations can be found in the paper.

![](https://user-images.githubusercontent.com/73837432/193831820-45c9d043-6d4d-4843-88de-171431fcb0f8.png)

# Explainability
The XAI plots can be obtained by running the cam_segmentation_precip.py and cam_segmentation_cloud_cover.py scripts. The code used was obtained at https://github.com/jacobgil/pytorch-grad-cam and adapted to our dataset and model.

![explanations](https://user-images.githubusercontent.com/73837432/193831432-ae138443-fa59-4c71-a00f-6c30b0d28d0a.png)
![cloud cover expl](https://user-images.githubusercontent.com/73837432/193831465-c49e4316-272f-4627-a559-d9ee3ae7fe1b.png)
