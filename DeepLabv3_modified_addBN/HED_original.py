import torch
from torch import Tensor
import torchvision
import torch.nn as nn
#from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        feature_channel: int = 256,
        feature_map_size_layer: int = 1,
        feature_accumumation_start: int = 1,
        multiLoss: bool = False,
        fix_kernel_size: bool = True,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.feature_channel = feature_channel
        self.feature_accumumation_start=feature_accumumation_start
        self.feature_map_size_layer=feature_map_size_layer
        self.multiLoss = multiLoss


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)


        #============================================
        #self.extra1 = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(4, 4), bias=False)
        #self.extra2 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #self.extra3 = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(4, 4), bias=False)

        #self.extra1 = nn.Conv2d(512, 2048, kernel_size=1, stride=1)
        #self.extra2 = nn.ConvTranspose2d(1024, 2048, kernel_size=1, stride=2)
        #self.extra3 = nn.ConvTranspose2d(2048, 2048, kernel_size=1, stride=4)


        #self.extra1 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        #self.extra2 = nn.ConvTranspose2d(1024, 1024, kernel_size=1, stride=2)
        #self.extra3 = nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=4)

        #previously used
        #if block==Bottleneck:
        #    self.extra1 = nn.Sequential(nn.Conv2d(512, self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))
        #    self.extra2 = nn.Sequential(nn.ConvTranspose2d(1024, self.feature_channel, kernel_size=1, stride=2),nn.ReLU(inplace=True))
        #    self.extra3 = nn.Sequential(nn.ConvTranspose2d(2048, self.feature_channel, kernel_size=1, stride=4),nn.ReLU(inplace=True))
        #else:
        #    self.extra1 = nn.Sequential(nn.Conv2d(128, self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))
        #    self.extra2 = nn.Sequential(nn.ConvTranspose2d(256, self.feature_channel, kernel_size=1, stride=2),nn.ReLU(inplace=True))
        #    self.extra3 = nn.Sequential(nn.ConvTranspose2d(512, self.feature_channel, kernel_size=1, stride=4),nn.ReLU(inplace=True))


        #self.extra4 = nn.BatchNorm2d(self.feature_channel)
        #end previously used




        #self.extra1 = nn.ConvTranspose2d(1024, 2048, kernel_size=1, stride=1)
        #self.extra2 = nn.ConvTranspose2d(2048, 2048, kernel_size=1, stride=2)
        #self.extra3 = nn.ConvTranspose2d(2048, 2048, kernel_size=1, stride=4)


        #self.extra2 = nn.Upsample(size = [2048,141, 169], mode='linear', align_corners=True)
        #self.extra3 = nn.Upsample(size = [2048,141, 169], mode='bilinear', align_corners=True)

        #self.conv1_2 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1_2 = norm_layer(self.inplanes)
        self.extra1 = self._make_extra_layer(block, 1)
        self.extra2 = self._make_extra_layer(block, 2)
        self.extra3 = self._make_extra_layer(block, 3)
        self.extra4 = self._make_extra_layer(block, 4)
        #self.extra5 = nn.BatchNorm2d(self.feature_channel)

        if self.feature_channel == 1:
            self.extra5 = nn.Sequential(nn.Conv2d(4, 1, kernel_size=1, stride=1),nn.ReLU(inplace=True))
        else:
            self.extra5 = nn.Sequential(nn.Conv2d(self.feature_channel*4, self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))

        #=====================================






        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_extra_layer(self, block: Type[Union[BasicBlock, Bottleneck]], current_layer):
        bottleneck_input_size = [256,512,1024,2048]
        baseblock_input_size = [64,128,256,512]
        #print(current_layer,self.feature_map_size_layer)
        if block == Bottleneck:
            input_size = bottleneck_input_size
        else:
            input_size = baseblock_input_size

        '''
        if self.feature_map_size_layer == current_layer:
            return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))
        if self.feature_map_size_layer > current_layer:
            if (self.feature_map_size_layer-current_layer) == 1:
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
            if (self.feature_map_size_layer-current_layer) == 2:
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(self.feature_channel, self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
            if (self.feature_map_size_layer-current_layer) == 3:
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(self.feature_channel, self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(self.feature_channel, self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
        if self.feature_map_size_layer < current_layer:
                stride = int((current_layer-self.feature_map_size_layer)*2)
                return nn.Sequential(nn.ConvTranspose2d(input_size[current_layer-1], self.feature_channel, kernel_size=1, stride=stride),nn.ReLU(inplace=True))

        '''
        if self.feature_channel == 1:
            if self.feature_map_size_layer == current_layer:
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], 1, kernel_size=1, stride=1),nn.ReLU(inplace=True))
            if self.feature_map_size_layer > current_layer:
                if (self.feature_map_size_layer-current_layer) == 1:
                    return nn.Sequential(nn.Conv2d(input_size[current_layer-1], 1, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
                if (self.feature_map_size_layer-current_layer) == 2:
                    return nn.Sequential(nn.Conv2d(input_size[current_layer-1], 1, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(input_size[current_layer-1], 1, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
                if (self.feature_map_size_layer-current_layer) == 3:
                    return nn.Sequential(nn.Conv2d(input_size[current_layer-1], 1, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(input_size[current_layer-1], 1, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(input_size[current_layer-1], 1, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
            if self.feature_map_size_layer < current_layer:
                stride = int(2**(current_layer-self.feature_map_size_layer))
                kernal_size = 2*stride
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], 1, kernel_size=1, stride=1),nn.ConvTranspose2d(1, 1, kernel_size=kernal_size, stride=stride),nn.ReLU(inplace=True))

        else:
            if self.feature_map_size_layer == current_layer:
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))
            if self.feature_map_size_layer > current_layer:
                if (self.feature_map_size_layer-current_layer) == 1:
                    return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
                if (self.feature_map_size_layer-current_layer) == 2:
                    return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
                if (self.feature_map_size_layer-current_layer) == 3:
                    return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
            if self.feature_map_size_layer < current_layer:
                #stride = int((current_layer-self.feature_map_size_layer)*2)
                stride = int(2**(current_layer-self.feature_map_size_layer))
                kernal_size = 2*stride
                return nn.Sequential(nn.Conv2d(input_size[current_layer-1], self.feature_channel, kernel_size=1, stride=1),nn.ConvTranspose2d(self.feature_channel, self.feature_channel, kernel_size=kernal_size, stride=stride),nn.ReLU(inplace=True))




        '''
        if block==Bottleneck:
            self.extra1 = nn.Sequential(nn.Conv2d(256, self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
            self.extra2 = nn.Sequential(nn.Conv2d(512, self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))
            self.extra3 = nn.Sequential(nn.ConvTranspose2d(1024, self.feature_channel, kernel_size=1, stride=2),nn.ReLU(inplace=True))
            self.extra4 = nn.Sequential(nn.ConvTranspose2d(2048, self.feature_channel, kernel_size=1, stride=4),nn.ReLU(inplace=True))
        else:
            self.extra1 = nn.Sequential(nn.Conv2d(64, self.feature_channel, kernel_size=3, stride=2, padding = 1),nn.ReLU(inplace=True))
            self.extra2 = nn.Sequential(nn.Conv2d(128, self.feature_channel, kernel_size=1, stride=1),nn.ReLU(inplace=True))
            self.extra3 = nn.Sequential(nn.ConvTranspose2d(256, self.feature_channel, kernel_size=1, stride=2),nn.ReLU(inplace=True))
            self.extra4 = nn.Sequential(nn.ConvTranspose2d(512, self.feature_channel, kernel_size=1, stride=4),nn.ReLU(inplace=True))


        self.extra5 = nn.BatchNorm2d(self.feature_channel)
        '''

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        if self.feature_accumumation_start == 1:
            x2 = self.extra1(x)
            h_w = (x2.size()[2],x2.size()[3])
            if self.multiLoss:
                out1 = x2
            #print(x2.size())

        x = self.layer2(x)
        if self.feature_accumumation_start == 2:
            x2 = self.extra2(x)
            h_w = (x2.size()[2],x2.size()[3])
            if self.multiLoss:
                out2 = x2
            
        elif self.feature_accumumation_start < 2:
            x22 = self.extra2(x)
            h_w = (x2.size()[2],x2.size()[3])
            x2 = torch.cat((x2,torchvision.transforms.functional.center_crop(x22,h_w)),1)

            if self.multiLoss:
                out2 = torchvision.transforms.functional.center_crop(x22,h_w)

        x = self.layer3(x)

        if self.feature_accumumation_start == 3:
            x2 = self.extra3(x)
            h_w = (x2.size()[2],x2.size()[3])
            if self.multiLoss:
                out3 = x2
        elif self.feature_accumumation_start < 3:
            x22 = self.extra3(x)
            x2 = torch.cat((x2,torchvision.transforms.functional.center_crop(x22,h_w)),1)

            if self.multiLoss:
                out3 = torchvision.transforms.functional.center_crop(x22,h_w)


        x = self.layer4(x)

        if self.feature_accumumation_start == 4:
            x2 = self.extra4(x)
            h_w = (x2.size()[2],x2.size()[3])
            if self.multiLoss:
                out4 = x2
        elif self.feature_accumumation_start < 4:
            x22 = self.extra4(x)
            x2 = torch.cat((x2,torchvision.transforms.functional.center_crop(x22,h_w)),1)

            if self.multiLoss:
                out4 = torchvision.transforms.functional.center_crop(x22,h_w)

        x2 = self.extra5(x2)

        if self.multiLoss:
      
            return x2,out1,out2,out3,out4
        else:
            return x2

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:


        #state_dict = load_state_dict_from_url(model_urls[arch],
        #                                      progress=progress)
        if arch != 'resnet18':
            print("not implemented")
            assert False
        resnet18_url = '/media/ignored/resnet18-5c106cde.pth' 
        model.load_state_dict(torch.load(resnet18_url),strict=False)
        print("weight loaded")
        
    return model


def resnet18(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
