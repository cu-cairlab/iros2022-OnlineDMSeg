import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch2trt import torch2trt


from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import cv2
import time
import argparse

from torch.nn import functional as F

import csv



from DM import DMData





def loadWeight_deeplabv3(model,classifier,weight):

    checkpoint = torch.load(weight)
    
    epoch = checkpoint['epoch']
    state_dict_model = {}
    state_dict_classifier = {}
    #model.load_state_dict(checkpoint['state_dict'])
    #print(checkpoint['state_dict'].keys())
    for key in checkpoint['state_dict'].keys():
        #print(key)
        key_split = key.split('.')
        if key_split[1] == "resnet":
            newkey = ".".join(key_split[2:])
            #print(newkey)
            state_dict_model[newkey] = checkpoint['state_dict'][key]
        elif key_split[1] == "aspp":
            newkey = ".".join(key_split[2:])
            #print(newkey)
            state_dict_classifier[newkey] = checkpoint['state_dict'][key]
        else:
            print("must wrong")
            assert False
    '''
    for name, param in model.named_parameters():
        print(name)


    for name, param in classifier.named_parameters():
        print(name)
    '''
    model.load_state_dict(state_dict_model)
    classifier.load_state_dict(state_dict_classifier)
    return model,classifier,epoch


def loadWeight(model,classifier,weight):

    checkpoint = torch.load(weight)
    
    epoch = checkpoint['epoch']
    state_dict_model = {}
    state_dict_classifier = {}
    #model.load_state_dict(checkpoint['state_dict'])
    #print(checkpoint['state_dict'].keys())
    for key in checkpoint['state_dict'].keys():
        #print(key)
        key_split = key.split('.')
        if key_split[1] == "backbone":
            newkey = ".".join(key_split[2:])
            #print(newkey)
            state_dict_model[newkey] = checkpoint['state_dict'][key]
        elif key_split[1] == "classifier":
            newkey = ".".join(key_split[2:])
            #print(newkey)
            state_dict_classifier[newkey] = checkpoint['state_dict'][key]
        else:
            print("must wrong")
            assert False
    '''
    for name, param in model.named_parameters():
        print(name)


    for name, param in classifier.named_parameters():
        print(name)
    '''
    model.load_state_dict(state_dict_model)
    classifier.load_state_dict(state_dict_classifier)
    return model,classifier,epoch





parser = argparse.ArgumentParser()

parser.add_argument('--weight', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--dataPath', type=str, default='',
                    help='path to data')

parser.add_argument('--calib_dataPath', type=str, default='DM',
                    help='path to data')


parser.add_argument('--dataset', type=str, default='DM',
                    help='path to data')

parser.add_argument('--net', type=str, default='DeepLabv3_modified_addBN',
                    help='path to data')


parser.add_argument('--trt', type=bool, default=True)
parser.add_argument('--trt_fp16Path', type=str, default='',
                    help='path to data')
parser.add_argument('--trt_int8Path', type=str, default='',
                    help='path to data')

parser.add_argument('--num_classes', type=int, default=4,
                    help='number of classes')

parser.add_argument('--backbone', type=str, default='resnet50',
                    help='resnet backbone')
parser.add_argument('--name_classifier', type=str, default='deeplab_modified',
                    help='resnet backbone')

parser.add_argument('--feature_channel', type=int, default=256,
                    help='new feature channel number')
parser.add_argument('--feature_map_size_layer', type=int, default=2,
                    help='new feature map size equals to the feature map size of layer [1,2,3,4]')
parser.add_argument('--feature_accumumation_start', type=int, default=2,
                    help='new feature accumulation start layer [1,2,3,4]')

parser.add_argument('--use_concat', type=bool, default=False)
parser.add_argument('--fix_kernel_size', type=bool, default=True)

args = parser.parse_args()




nb_classes = args.num_classes
backbone = args.backbone
if True:
  if args.net == 'DeepLabv3_modified_addBN':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,
                   feature_map_size_layer=args.feature_map_size_layer,feature_accumumation_start=args.feature_accumumation_start,
                   pretrained=False,concat = args.use_concat,name_classifier=args.name_classifier,fix_kernel_size=args.fix_kernel_size)
      is_seperate_classifier = False


  elif args.net == 'HED':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,
                 feature_map_size_layer=args.feature_map_size_layer,feature_accumumation_start=args.feature_accumumation_start,pretrained=False,original=True,name_classifier=args.name_classifier)
      is_seperate_classifier = False



  elif args.net == 'Deeplabv3':
      from deeplabv3_model import deeplabv3
      model = deeplabv3.DeepLabV3(nb_classes,backbone=args.backbone,name_classifier=args.name_classifier,pretrained=False)


      #from DeepLabv3_modified_addBN import custom_net
      #model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel)
      is_seperate_classifier = False


  elif args.net == 'SFSegNet':

      #from network.sfnet_resnet import DeepR18_SF_deeply
      #model = DeepR18_SF_deeply(nb_classes, criterion=None)
      if args.backbone == 'resnet':
          from network.sfnet_resnet import DeepR18_SF_deeply
          model = DeepR18_SF_deeply(nb_classes, criterion=None)
      else:
          from network.sfnet_dfnet import AlignedDFnetv2_FPNDSN
          model = AlignedDFnetv2_FPNDSN(nb_classes, criterion=None)


      is_seperate_classifier = False
  elif args.net == 'Attanet':

      #from network.sfnet_resnet import DeepR18_SF_deeply
      #model = DeepR18_SF_deeply(nb_classes, criterion=None)
      from Attanet import AttaNet
      model = AttaNet.AttaNet(nb_classes,pretrained=False)
      is_seperate_classifier = False

  else:
      print(args.net)
      print("net not defined")




  if os.path.isfile(args.weight):
    print('=> loading checkpoint {0}'.format(args.weight))
    if args.net == 'Deeplabv3' and is_seperate_classifier:
        model,classifier,epoch = loadWeight_deeplabv3(model,classifier,args.weight)
    elif is_seperate_classifier:
        model,classifier,epoch = loadWeight(model,classifier,args.weight)
    else:
        checkpoint = torch.load(args.weight)
        epoch = checkpoint['epoch']
        state_dict_model = {}
        for key in checkpoint['state_dict'].keys():
            key_split = key.split('.')

            newkey = ".".join(key_split[1:])
            #print(newkey)
            state_dict_model[newkey] = checkpoint['state_dict'][key]

        model.load_state_dict(state_dict_model)
    '''
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    '''
    print('=> loaded checkpoint {0} (epoch {1})'.format(
      args.weight, epoch))

  else:
    print('=> no checkpoint found at {0}'.format(args.weight))








#w = int(1125)
#h = int(1352)


w = int(1500)
h = int(2048)


bs = int(1)

print([w,h])
print((int(3376/h))*(int(2704/w)))


input_data = torch.rand(bs,3, w, h).cuda()
model = model.cuda()
model.eval()

model.half()
if is_seperate_classifier:
    classifier = classifier.cuda().eval().half()




#from ImageMask2Tiles_uniform import processImg
#from ImageMask2Tiles_uniform import assemblyImg




fp = 'reference.jpg'


print("+++++++ trt fp16 +++++++++++")


input_sample = torch.rand(bs,3, w, h).cuda().half()
model_trt = torch2trt(model, [input_sample],max_batch_size=bs,fp16_mode=True)
torch.save(model_trt.state_dict(), args.trt_fp16Path)

from torch2trt import TRTModule
model_trt = TRTModule()

model_trt.load_state_dict(torch.load(args.trt_fp16Path))



print("------loaded--------")


print("+++++++ trt int8 +++++++++++")

calib_dataset = DMData(args.calib_dataPath,train=False,test=True)


input_sample = torch.rand(bs,3, w, h).cuda().half()
model_trt = torch2trt(model, [input_sample],max_batch_size=bs,int8_mode=True,int8_calib_dataset=calib_dataset)
torch.save(model_trt.state_dict(), args.trt_int8Path)

from torch2trt import TRTModule
model_trt = TRTModule()

model_trt.load_state_dict(torch.load(args.trt_int8Path))
torch.cuda.synchronize()
print("------loaded--------")



