import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models.segmentation import segmentation as segmentation

from pascal import VOCSegmentation
from cityscapes import Cityscapes
from DM import DMData
from utils import AverageMeter, inter_and_union
import cv2

import csv
import time


parser = argparse.ArgumentParser()

parser.add_argument('--weight', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--dataPath', type=str, default='',
                    help='path to data')
parser.add_argument('--dataset', type=str, default='DM',
                    help='path to data')

parser.add_argument('--output', type=str, default='DM',
                    help='path to data')


parser.add_argument('--net', type=str, default='DeepLabv3_modified_addBN',
                    help='path to data')


parser.add_argument('--num_classes', type=int, default=4,
                    help='number of classes')

parser.add_argument('--backbone', type=str, default='resnet50',
                    help='resnet backbone')


parser.add_argument('--feature_channel', type=int, default=256,
                    help='new feature channel number')




args = parser.parse_args()


def main():
  assert torch.cuda.is_available()
  nb_classes = args.num_classes
  torch.backends.cudnn.benchmark = True
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit')
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes')
  elif args.dataset == 'DM':
    dataset = DMData(args.dataPath,train=False,test=True)
    nb_classes = len(dataset.CLASSES)+1
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))




  #get net


  if args.net == 'DeepLabv3_modified_addBN':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel)


  elif args.net == 'DeepLabv3_modified':

      from DeepLabv3_modified import custom_net
      model=custom_net.Net(nb_classes)

  elif args.net == 'DeepLabv3_original':
      from torchvision.models.segmentation import segmentation
      model = segmentation._load_model('deeplabv3', 'resnet50', pretrained=False, progress=True, num_classes=nb_classes, aux_loss=False)

      model.classifier[4]=nn.Conv2d(
        in_channels=256,
        out_channels=nb_classes,
        kernel_size=1,
        stride=1
      )

      model.aux_classifier = None

  elif args.net == 'DeepLabv3_free':

      from DeepLabv3_free import deeplabv3 as custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone)

  elif args.net == 'FCN':

      from FCN import fcn as custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone)

  elif args.net == 'SFSegNet':

      #from network.sfnet_resnet import DeepR18_SF_deeply
      #model = DeepR18_SF_deeply(nb_classes, criterion=None)
      from network.sfnet_dfnet import AlignedDFnetv2_FPNDSN
      model = AlignedDFnetv2_FPNDSN(nb_classes, criterion=None)
  else:
      print(args.net)
      print("net not defined")



  weights = [1., 1.0,0]
  class_weights = torch.FloatTensor(weights).cuda()
  #criterion = nn.CrossEntropyLoss(ignore_index=255,weight=class_weights)
  model = nn.DataParallel(model).cuda()
  optimizer = optim.Adam(model.parameters())

  if os.path.isfile(args.weight):
    print('=> loading checkpoint {0}'.format(args.weight))
    checkpoint = torch.load(args.weight)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint {0} (epoch {1})'.format(
      args.weight, checkpoint['epoch']))
  else:
    print('=> no checkpoint found at {0}'.format(args.weight))
  with torch.no_grad():
    validation(model,dataset,epoch=1,folder=args.output)


def validation(model,dataset,epoch,folder=None):
    model.eval()
    if folder!= None:
      if not os.path.exists(folder):
          os.mkdir(folder)
      cmap = loadmat('pascal_seg_colormap.mat')['colormap']
      cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for i in range(len(dataset)):      
      inputs, target,_ = dataset[i]

      img = (inputs.permute(1, 2, 0).data.numpy()*255).squeeze().astype(np.int32)
      mask = target.numpy().astype(np.uint8)
      inputs = Variable(inputs.cuda())

      outputs = model(inputs.unsqueeze(0))#['out']
      #torch.cuda.synchronize()



      if args.net == 'DeepLabv3_original':
          outputs = outputs["out"]


      #time.sleep(100)
      start = time.time()

      _, pred = torch.max(outputs, 1)



      pred = pred.data.cpu()



      pred = pred.numpy().squeeze().astype(np.uint8)

      print(np.unique(pred))

      imname = dataset.masks[i].split('/')[-1]

      redImg = np.zeros(img.shape, img.dtype)
      redImg[:,:] = (0, 0, 255)
      leafMask = np.copy(pred)
      leafMask[leafMask==1] = 0
      leafMask[leafMask==2] = 1
      redMask = cv2.bitwise_and(redImg, redImg, mask=leafMask)



      blueImg = np.zeros(img.shape, img.dtype)
      blueImg[:,:] = (0, 255, 255)
      DMMask = np.copy(pred)
      DMMask[DMMask==2] = 0
      blueImg = cv2.bitwise_and(blueImg, blueImg, mask=DMMask)



      #mask_pred.save(os.path.join(folder, imname))

      img = img.copy().astype(np.float32)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB ).astype(np.int32)
      cv2.addWeighted(redMask, 0.25, img, 1, 0, img)
      cv2.addWeighted(blueImg, 0.25, img, 1, 0, img)
      cv2.imwrite(os.path.join(folder, imname),img)




      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
    #logwriter.writerow([epoch+1,iou[0]*100,iou[1]*100,iou.mean()*100])  
    model.train()

if __name__ == "__main__":
  main()
