import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import random
import cv2

from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models.segmentation import segmentation as segmentation

from pascal import VOCSegmentation
from cityscapes import Cityscapes
from DM import DMData
from utils import AverageMeter, inter_and_union
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torchvision.models.segmentation.fcn import FCNHead #for training with multi loss
import csv


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


def loadWeight_HED_multiLoss(model,classifier,weight):

    checkpoint = torch.load(weight)
    
    epoch = checkpoint['epoch']
    state_dict_model = {}
    state_dict_classifier = {}
    #model.load_state_dict(checkpoint['state_dict'])
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
parser.add_argument('--img', type=str, default='/media/Data/row001_auto/color/cam0_images/',
                    help='path to data')
parser.add_argument('--output', type=str, default='DM',
                    help='path to data')

parser.add_argument('--net', type=str, default='DeepLabv3_modified_addBN',
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
parser.add_argument('--multiLoss', type=bool, default=False)




args = parser.parse_args()



nb_classes = args.num_classes
backbone = args.backbone
classifier=None


def drawMask(pred,output_dir,name):
    leafMask = np.copy(pred)
    leafMask[leafMask==1] = 0
    leafMask[leafMask==2] = 1
    leafMask[leafMask==3] = 0
    DMMask = np.copy(pred)
    DMMask[DMMask==2] = 0
    DMMask[DMMask==3] = 0
    blueImg = np.zeros((DMMask.shape[0],DMMask.shape[1],3),dtype = np.uint8)
    redImg = np.zeros((DMMask.shape[0],DMMask.shape[1],3),dtype = np.uint8)
    blueImg[:,:] = (0, 128, 0)
    blueImg = cv2.bitwise_and(blueImg, blueImg, mask=leafMask)

    redImg[:,:] = (0, 0, 255)
    redImg = cv2.bitwise_and(redImg, redImg, mask=DMMask)
    blueImg = blueImg + redImg
    print(os.path.join(output_dir,name))
    cv2.imwrite(os.path.join(output_dir,name),blueImg)

if True:
  if args.net == 'DeepLabv3_modified_addBN':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,
                   feature_map_size_layer=args.feature_map_size_layer,feature_accumumation_start=args.feature_accumumation_start,pretrained=False,concat = args.use_concat,name_classifier=args.name_classifier)
      is_seperate_classifier = False


  elif args.net == 'HED':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,feature_map_size_layer=args.feature_map_size_layer,
                             feature_accumumation_start=args.feature_accumumation_start,pretrained=False,original=True,multiLoss=args.multiLoss,name_classifier=args.name_classifier)

      if args.multiLoss:
          aux_head1 = FCNHead(64,nb_classes).cuda()
          aux_head2 = FCNHead(64,nb_classes).cuda()
          aux_head3 = FCNHead(64,nb_classes).cuda()
          aux_head4 = FCNHead(64,nb_classes).cuda()


      is_seperate_classifier = False




  elif args.net == 'Deeplabv3':
      from deeplabv3_model.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
      if backbone=="resnet18":
        model=ResNet18_OS16()
      elif backbone=="resnet18OS8":
        model=ResNet18_OS8()
      elif backbone=="resnet34":
        model=ResNet34_OS16()
      elif backbone=="resnet50":
        model=ResNet50_OS16()
      
      from torchvision.models.segmentation.deeplabv3 import DeepLabHead
      classifier = DeepLabHead(512,nb_classes)


      #from DeepLabv3_modified_addBN import custom_net
      #model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel)
      is_seperate_classifier = True


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
    print("loading weight")
    checkpoint = torch.load(args.weight)
    start_epoch = checkpoint['epoch']



    state_dict_model = {}
    for key in checkpoint['state_dict'].keys():
       key_split = key.split('.')

       newkey = ".".join(key_split[1:])
            #print(newkey)
       state_dict_model[newkey] = checkpoint['state_dict'][key]

    model.load_state_dict(state_dict_model)



    #optimizer.load_state_dict(checkpoint['optimizer'])

    if args.multiLoss:
      print("loading multi head weight")
      '''
      state_dict_aux_head1 = {}
      for key in checkpoint['aux_head1'].keys():
         key_split = key.split('.')

         newkey = ".".join(key_split[1:])
            #print(newkey)
         state_dict_aux_head1[newkey] = checkpoint['aux_head1'][key]

      state_dict_aux_head2 = {}
      for key in checkpoint['aux_head2'].keys():
         key_split = key.split('.')

         newkey = ".".join(key_split[1:])
            #print(newkey)
         state_dict_aux_head2[newkey] = checkpoint['aux_head2'][key]

      state_dict_aux_head3 = {}
      for key in checkpoint['aux_head3'].keys():
         key_split = key.split('.')

         newkey = ".".join(key_split[1:])
            #print(newkey)
         state_dict_aux_head3[newkey] = checkpoint['aux_head4'][key]

      state_dict_aux_head4 = {}
      for key in checkpoint['aux_head4'].keys():
         key_split = key.split('.')

         newkey = ".".join(key_split[1:])
            #print(newkey)
         state_dict_aux_head4[newkey] = checkpoint['aux_head4'][key]


      aux_head1.load_state_dict(state_dict_aux_head1)
      aux_head2.load_state_dict(state_dict_aux_head2)
      aux_head3.load_state_dict(state_dict_aux_head3)
      aux_head4.load_state_dict(state_dict_aux_head4)
      '''

      aux_head1.load_state_dict(checkpoint['aux_head1'])
      aux_head2.load_state_dict(checkpoint['aux_head2'])
      aux_head3.load_state_dict(checkpoint['aux_head3'])
      aux_head4.load_state_dict(checkpoint['aux_head4'])

    print("loaded")

  model = model.cuda().half()
  model.eval()
  if args.multiLoss:
    aux_head1 = aux_head1.half()
    aux_head2 = aux_head2.half()
    aux_head3 = aux_head3.half()
    aux_head4 = aux_head4.half()

  nb_classes = 4


  data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
 
  fp = args.img
  fn = fp.split('/')[-1]
  print(fp)
  print(fn)
  input_img = np.asarray(Image.open(fp).convert('RGB'))
  input_data = data_transforms(input_img).unsqueeze_(0).cuda().half()

  def processOutput(outputs,output_dir,name):
    print("output:")
    print(outputs.size())


    outputs = outputs[:,:3,:,:]

    outputs_std = torch.std(outputs,1).squeeze()

    print("output_std:", outputs_std.size())
    


    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu()
    pred = pred.numpy().squeeze().astype(np.uint8)
    outputs = outputs.data.cpu().numpy()
    outputs_std = outputs_std.data.cpu().numpy()
    print(np.unique(outputs))
    print(np.unique(outputs_std))
    outputs_std_masks = outputs_std<0.5
    print(pred.shape)
    print(outputs_std_masks.shape)
    #pred[outputs_std_masks] = 0




    drawMask(pred,output_dir,name)
    '''

    leafMask = np.copy(pred)
    leafMask[leafMask==1] = 0
    leafMask[leafMask==2] = 1
    leafMask[leafMask==3] = 0
    DMMask = np.copy(pred)
    DMMask[DMMask==2] = 0
    DMMask[DMMask==3] = 0
    blueImg = np.zeros((DMMask.shape[0],DMMask.shape[1],3),dtype = np.uint8)
    redImg = np.zeros((DMMask.shape[0],DMMask.shape[1],3),dtype = np.uint8)
    blueImg[:,:] = (0, 128, 0)
    blueImg = cv2.bitwise_and(blueImg, blueImg, mask=leafMask)

    redImg[:,:] = (0, 0, 255)
    redImg = cv2.bitwise_and(redImg, redImg, mask=DMMask)
    blueImg = blueImg + redImg
    print(os.path.join(output_dir,name))
    cv2.imwrite(os.path.join(output_dir,name),blueImg)
    '''
    

  if args.net == 'HED' and args.multiLoss:
    #target_aux = target
    out,out1,out2,out3,out4 = model(input_data)
    print(input_data.size())
    out1 = aux_head1(out1)
    out2 = aux_head2(out2)
    out3 = aux_head3(out3)
    out4 = aux_head4(out4)
    out1 = nn.Upsample((input_data.size()[2], input_data.size()[3]), mode='bilinear', align_corners=False)(out1)
    out2 = nn.Upsample((input_data.size()[2], input_data.size()[3]), mode='bilinear', align_corners=False)(out2)
    out3 = nn.Upsample((input_data.size()[2], input_data.size()[3]), mode='bilinear', align_corners=False)(out3)
    out4 = nn.Upsample((input_data.size()[2], input_data.size()[3]), mode='bilinear', align_corners=False)(out4)
    processOutput(out,args.output,fn[:-4]+'.png')
    processOutput(out1,args.output,fn[:-4]+'_1.png')
    processOutput(out2,args.output,fn[:-4]+'_2.png')
    processOutput(out3,args.output,fn[:-4]+'_3.png')
    processOutput(out4,args.output,fn[:-4]+'_4.png')
  else:
    out = model(input_data)
    processOutput(out,args.output,fn[:-4]+'.png')
  out = cv2.imread(fp[:-4]+'.png',0)
  drawMask(out,args.output,fn[:-4]+'_gt.png')
  

