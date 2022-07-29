# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
#from aspp import ASPP, ASPP_Bottleneck
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=4,backbone="resnet50",pretrained=True,name_classifier='deeplab_modified'):
        super(DeepLabV3, self).__init__()
        if name_classifier=='deeplab_modified':
            from . import deeplabv3_head
            DeepLabHead = deeplabv3_head.DeepLabHead
        elif name_classifier == 'deeplab':
            from torchvision.models.segmentation.deeplabv3 import DeepLabHead
        else:
            print('classifier not defined')
            assert False

        if backbone=="resnet18":
            model=ResNet18_OS16(pretrained)
        elif backbone=="resnet18OS8":
            model=ResNet18_OS8(pretrained)
        elif backbone=="resnet34":
            model=ResNet34_OS16(pretrained)
        elif backbone=="resnet50":
            model=ResNet50_OS16(pretrained)


        self.num_classes = num_classes
        self.resnet = model # NOTE! specify the type of ResNet here
        self.aspp = DeepLabHead(512,self.num_classes) #ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output
