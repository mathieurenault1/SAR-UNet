# SAR-UNet
Code for the Paper "SAR-UNet: Small Attention Residual UNet for Explainable Precipitation Nowcasting". [(ArXiv link)](https://arxiv.org/abs/2303.06663)
Accepted for publication at the [International Joint Conference on Neural Networks (IJCNN 2023)](https://2023.ijcnn.org/)

![sar unet](https://user-images.githubusercontent.com/73837432/193812363-4e9a817d-fd9e-47de-a621-e32766258d0a.png)

# Datasets & Models
If you want access to the datasets (precipitation maps & cloud cover dataset) used in this paper, please visit https://github.com/HansBambel/SmaAt-UNet for further details.
Please put the dataset into "\data\precip" directory for training and testing the precipitation network, and put the trained models into "\checkpoints\eval\precip" directory for future loading.

# Training
For training on the precipitation task we used the train_precip_lightning.py file.
For training on the cloud cover task we used the train_cloud_cover.py file.
Training was done using Pytorch-Lightning. All details about parameters used and device configurations can be found in the paper.

![prediction example2](https://user-images.githubusercontent.com/73837432/193834072-aa05eb3d-14a5-4c17-bc12-1b51daac7e06.jpg)

# Explainability
The XAI plots can be obtained by running the cam_segmentation_precip.py and cam_segmentation_cloud_cover.py scripts. The code used was obtained at https://github.com/jacobgil/pytorch-grad-cam and adapted to our dataset and model.
![explanations](https://user-images.githubusercontent.com/73837432/193834255-37c54c38-69ae-47b1-9ebd-588dd4b476ad.jpg)
![cloud cover expl (1)](https://user-images.githubusercontent.com/73837432/193833856-1f882654-3696-47f8-bfae-1632fd51cade.jpg)

# Citation
If you use our data and code, please cite the paper using the following bibtex reference:
```
@misc{https://doi.org/10.48550/arxiv.2303.06663,
  doi = {10.48550/ARXIV.2303.06663},
  url = {https://arxiv.org/abs/2303.06663},
  author = {Renault, Mathieu and Mehrkanoon, Siamak},
  keywords = {Machine Learning (cs.LG), Atmospheric and Oceanic Physics (physics.ao-ph), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences, I.2; I.5},
  title = {SAR-UNet: Small Attention Residual UNet for Explainable Nowcasting Tasks},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```

