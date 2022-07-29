import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
#from resnet50 import Resnet50 as backbone
from . import resnet_custom as resnet

from torchvision.models.segmentation.deeplabv3 import DeepLabHead as DeepLabHead_original
from torchvision.models.segmentation.fcn import FCNHead as FCNHead_original
from . import deeplabv3
from . import FCNHead
#classifier = deeplabv3.DeepLabHead
      #classifier = deeplabv3.DeepLabHead(args.feature_channel,nb_classes)

class Net(nn.Module):

    def __init__(self,num_classes=3,backbone="resnet50",feature_channel=256,feature_map_size_layer=2,feature_accumumation_start=2,pretrained=True,concat=True,original=False,multiLoss=False,name_classifier='deeplab_modified',fix_kernel_size=True):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.multiLoss=multiLoss

        if name_classifier == 'deeplab_modified':
            classifier = deeplabv3.DeepLabHead
        elif name_classifier == 'FCN_original':
            classifier = FCNHead_original
        elif name_classifier == 'deeplab':
            classifier = DeepLabHead_original
        elif name_classifier == 'FCN':
            classifier = FCNHead.FCNHead
        else:
            print('classifier not defined')
            assert False


        if (not concat) and (not original):
            from . import resnet_custom as resnet
            self.classifier = classifier(feature_channel,num_classes)
        if concat:
            from . import resnet_custom_concat as resnet
            self.classifier = classifier(feature_channel*4,num_classes)
        if original:
            from . import HED_original as resnet
            feature_map_size_layer = 1
            feature_accumumation_start = 1
            self.classifier = classifier(feature_channel,num_classes)


        print('feature_map_size_layer',feature_map_size_layer)
        print('feature_accumumation_start',feature_accumumation_start)
        if backbone=="resnet50":
            self.backbone=resnet.resnet50(replace_stride_with_dilation = [False, False, False],feature_channel=feature_channel,feature_map_size_layer=feature_map_size_layer,feature_accumumation_start=feature_accumumation_start)
        if backbone=="resnet34":
            self.backbone=resnet.resnet34(replace_stride_with_dilation = [False, False, False],feature_channel=feature_channel,feature_map_size_layer=feature_map_size_layer,feature_accumumation_start=feature_accumumation_start)
        if backbone=="resnet18":
            self.backbone=resnet.resnet18(replace_stride_with_dilation = [False, False, False],feature_channel=feature_channel,feature_map_size_layer=feature_map_size_layer,feature_accumumation_start=feature_accumumation_start,pretrained=pretrained,multiLoss=multiLoss,fix_kernel_size = fix_kernel_size)

        
        #self.classifier = classifier(feature_channel,num_classes)
        
        #self.classifier = classifier(512,3)




    def forward(self, x):
        # Max pooling over a (2, 2) window
        size = (x.shape[2], x.shape[3])
        if self.multiLoss:
            x,out1,out2,out3,out4 = self.backbone(x)
        else:
            x = self.backbone(x)
        x = self.classifier(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=False)(x)
        if self.multiLoss:
            return x,out1,out2,out3,out4
        else:
            return x

if __name__=="__main__":
    net = Net()
    print(net)

