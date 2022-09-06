import argparse
from models.unet_parts import *
from models.DS_parts import *
from models.layers import *
from models.RR_parts import *
import pytorch_lightning as pl
from models.regression_lightning import Precip_regression_base
from cloud_cover.cloud_cover_lightning import Cloud_base


class UNet_precip(Precip_regression_base):
    def __init__(self, hparams):
        super(UNet_precip, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.inc = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SmaAt_UNet_precip(Precip_regression_base):
    def __init__(self, hparams,  kernels_per_layer=1, reduction_ratio = 16):
        super(SmaAt_UNet_precip, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.inc = DoubleConvDS(self.in_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class SAR_UNet_precip(Precip_regression_base):
    def __init__(self, hparams, kernels_per_layer=1, reduction_ratio = 16):
        super(SAR_UNet_precip, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.RRCNN1 = ResDoubleConvDS(in_channels=self.in_channels, out_channels=64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.RRCNN2 = ResDoubleConvDS(in_channels=64, out_channels=128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.RRCNN3 = ResDoubleConvDS(in_channels=128, out_channels=256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.RRCNN4 = ResDoubleConvDS(in_channels=256, out_channels=512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        self.RRCNN5 = ResDoubleConvDS(in_channels=512, out_channels=1024, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024, reduction_ratio=reduction_ratio)

        self.Up5 = UpDS_Simple(1024, 512, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN5 = ResDoubleConvDS(in_channels=1024, out_channels=512)
        self.Up4 = UpDS_Simple(512, 256, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN4 = ResDoubleConvDS(in_channels=512, out_channels=256)
        self.Up3 = UpDS_Simple(256, 128, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN3 = ResDoubleConvDS(in_channels=256, out_channels=128)
        self.Up2 = UpDS_Simple(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN2 = ResDoubleConvDS(in_channels=128, out_channels=64)
        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.RRCNN1(x)
        x1 = self.cbam1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        x2 = self.cbam2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        x3 = self.cbam3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        x4 = self.cbam4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        x5 = self.cbam5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        logits = self.outc(d2)
        return logits

class SmaAt_UNet_cloud(Cloud_base):
    def __init__(self, hparams,  kernels_per_layer=1, reduction_ratio = 16):
        super(SmaAt_UNet_cloud, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.inc = DoubleConvDS(self.in_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class SAR_UNet_cloud(Cloud_base):
    def __init__(self, hparams, kernels_per_layer=1, reduction_ratio = 16):
        super(SAR_UNet_cloud, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.RRCNN1 = ResDoubleConvDS(in_channels=self.in_channels, out_channels=64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.RRCNN2 = ResDoubleConvDS(in_channels=64, out_channels=128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.RRCNN3 = ResDoubleConvDS(in_channels=128, out_channels=256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.RRCNN4 = ResDoubleConvDS(in_channels=256, out_channels=512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        self.RRCNN5 = ResDoubleConvDS(in_channels=512, out_channels=1024, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024, reduction_ratio=reduction_ratio)

        self.Up5 = UpDS_Simple(1024, 512, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN5 = ResDoubleConvDS(in_channels=1024, out_channels=512)
        self.Up4 = UpDS_Simple(512, 256, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN4 = ResDoubleConvDS(in_channels=512, out_channels=256)
        self.Up3 = UpDS_Simple(256, 128, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN3 = ResDoubleConvDS(in_channels=256, out_channels=128)
        self.Up2 = UpDS_Simple(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.Up_RRCNN2 = ResDoubleConvDS(in_channels=128, out_channels=64)
        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.RRCNN1(x)
        x1 = self.cbam1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        x2 = self.cbam2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        x3 = self.cbam3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        x4 = self.cbam4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        x5 = self.cbam5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        logits = self.outc(d2)
        return logits
