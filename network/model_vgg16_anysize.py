import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)

        vgg16_bn.features[0] = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.vgg_features = vgg16_bn.features
        self.avg_pool = vgg16_bn.avgpool

        self.layer1 = vgg16_bn.features[0:6]
        self.layer2 = vgg16_bn.features[6:13]
        self.layer3 = vgg16_bn.features[13:23]
        self.layer4 = vgg16_bn.features[23:33]
        self.layer5 = vgg16_bn.features[33:43]

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        return c1, c2, c3, c4, c5


class DecoderBlock(nn.Module):
    def __init__(self, channel):
        super(DecoderBlock, self).__init__()

        self.convlayer1 = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(channel // 2)
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(channel // 2)
        )

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)

        return x


class Deconvolution(nn.Module):
    def __init__(self, channel):
        super(Deconvolution, self).__init__()
        self.deconv = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2, bias=False)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = self.deconv(x)
        x = self.prelu(x)
        x = self.bn(x)

        return x


class CoarseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoderblock1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(512),
        )
        self.deconvolution1 = Deconvolution(512)

        self.decoderblock2 = DecoderBlock(512)
        self.deconvolution2 = Deconvolution(256)

        self.decoderblock3 = DecoderBlock(256)
        self.deconvolution3 = Deconvolution(128)

        self.decoderblock4 = DecoderBlock(128)
        self.deconvolution4 = Deconvolution(64)

        self.decoderblock5 = DecoderBlock(64)
        self.CoarseConvP = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c1, c2, c3, c4, c5):
        x = self.decoderblock1(c5)
        x = self.deconvolution1(x)

        if x.shape[2:] != c4.shape[2:]:
            x = nn.functional.interpolate(x, size=c4.size()[2:], mode='bilinear')
        x = c4 + x  # skip connection
        x = self.decoderblock2(x)
        x = self.deconvolution2(x)

        if x.shape[2:] != c3.shape[2:]:
            x = nn.functional.interpolate(x, size=c3.size()[2:], mode='bilinear')
        x = c3 + x
        x = self.decoderblock3(x)
        x = self.deconvolution3(x)

        if x.shape[2:] != c2.shape[2:]:
            x = nn.functional.interpolate(x, size=c2.size()[2:], mode='bilinear')
        x = c2 + x
        x = self.decoderblock4(x)
        x = self.deconvolution4(x)

        if x.shape[2:] != c1.shape[2:]:
            x = nn.functional.interpolate(x, size=c1.size()[2:], mode='bilinear')
        x = c1 + x
        x = self.decoderblock5(x)
        x = self.CoarseConvP(x)
        x = self.sigmoid(x)

        return x


class AtrousBlock(nn.Module):
    def __init__(self, atrous_rate):
        super(AtrousBlock, self).__init__()
        self.atrous1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32)
        )
        self.atrous2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32)
        )
        self.atrous3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(16)
        )

    def forward(self, x):
        x = self.atrous1(x)
        x = self.atrous2(x)
        x = self.atrous3(x)

        return x


class FineDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atrousblock1 = AtrousBlock(atrous_rate=1)
        self.atrousblock2 = AtrousBlock(atrous_rate=2)
        self.atrousblock3 = AtrousBlock(atrous_rate=3)

        self.AsppPooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 3, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(3, 48, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(16)
        )

        self.FineConvP = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, iact, coarse_out):
        input = torch.cat([data, iact, coarse_out], dim=1)

        x1 = self.atrousblock1(input)
        x2 = self.atrousblock2(input)
        x3 = self.atrousblock3(input)
        x = torch.cat([x1, x2, x3], dim=1)

        x4 = self.AsppPooling(x)
        out = x4 * x
        out = self.project(out)
        out = self.FineConvP(out)
        out = self.sigmoid(out)

        return out


class Model_BRS(nn.Module):
    def __init__(self):
        super(Model_BRS, self).__init__()
        self.encoder = Encoder()
        self.coarse_decoder = CoarseDecoder()
        self.fine_decoder = FineDecoder()
        self.withFD = False

    def forward(self, data, iact):
        x = torch.cat([data, iact], dim=1)
        c1, c2, c3, c4, c5 = self.encoder(x)
        coarse_out = self.coarse_decoder(c1, c2, c3, c4, c5)

        # first 20 epochs: train without the fine decoder, remaining 15 epochs: train with the fine decoder
        if self.withFD == False:
            return coarse_out
        else:
            fine_out = self.fine_decoder(data, iact, coarse_out)
            return fine_out
