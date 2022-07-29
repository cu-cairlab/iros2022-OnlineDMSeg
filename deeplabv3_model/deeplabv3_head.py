import torch
from torch import nn
from torch.nn import functional as F

#from ._utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]




class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=32):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        #print(rate)
        #print(self.convs)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.pool = ASPPPooling(in_channels, out_channels)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conv2 =nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        #self.dconv1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),nn.ReLU(inplace=True))
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=2),nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=4),nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=8),nn.ReLU(inplace=True))
        print("using new head")

    def forward(self, x):
        res = []

        res.append(self.pool(x))
        x = self.conv0(x)
        x2 = x
        #print(x2.size())
        res.append(x2)
        x = self.conv1(x)
        x22 = self.dconv1(x)
        #x22 = nn.functional.pad(x22,(0,x2.size()[3]-x22.size()[3],0,x2.size()[2]-x22.size()[2]))
        #print(x22.size())
        res.append(nn.functional.pad(x22,(0,x2.size()[3]-x22.size()[3],0,x2.size()[2]-x22.size()[2])))
        x = self.conv2(x)
        x22 = self.dconv2(x)
        #x22 = nn.functional.pad(x22,(0,x2.size()[3]-x22.size()[3],0,x2.size()[2]-x22.size()[2]))
        #print(x22.size())
        res.append(nn.functional.pad(x22,(0,x2.size()[3]-x22.size()[3],0,x2.size()[2]-x22.size()[2])))
        x = self.conv3(x)
        x22 = self.dconv3(x)
        #print(x22.size())
        #print(x2.size()[2]-x22.size()[2])
        #print(x2.size()[3]-x22.size()[3])
        #x22 = nn.functional.pad(x22,(0,x2.size()[3]-x22.size()[3],0,x2.size()[2]-x22.size()[2]))
        #print(x22.size())
        res.append(nn.functional.pad(x22,(0,x2.size()[3]-x22.size()[3],0,x2.size()[2]-x22.size()[2])))  
        





        '''
        for conv in self.convs:
            print(conv(x).size())
            res.append(conv(x))
        '''
        res = torch.cat(res, dim=1)
        return self.project(res)
