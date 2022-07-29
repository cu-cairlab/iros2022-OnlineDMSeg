import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
#from resnet50 import Resnet50 as backbone
from . import resnet

from torchvision.models.segmentation.deeplabv3 import DeepLabHead as classifier
#from torchvision.models.segmentation.fcn import FCNHead as classifier


class Net(nn.Module):

    def __init__(self,num_classes=3):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        if backbone=="resnet50":
            self.backbone=resnet.resnet50(replace_stride_with_dilation = [False, False, False],pretrained=True)
        if backbone=="resnet34":
            self.backbone=resnet.resnet34(replace_stride_with_dilation = [False, False, False],pretrained=True)
        self.classifier = classifier(2048,num_classes)

        #self.classifier = classifier(512,3)




    def forward(self, x):
        # Max pooling over a (2, 2) window
        size = (x.shape[2], x.shape[3])
        x = self.backbone(x)
        x = self.classifier(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x


if __name__=="__main__":
    net = Net()
    print(net)

