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

import custom_net
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
args = parser.parse_args()


def main():
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit')
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes')
  elif args.dataset == 'DM':
    dataset = DMData(args.dataPath,train=False,test=True)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  '''
  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))
  '''
  nb_classes = 3
  '''
  model = segmentation.deeplabv3_resnet101(pretrained=False,num_classes=21)
  model.classifier[4]=nn.Conv2d(
    in_channels=256,
    out_channels=nb_classes,
    kernel_size=1,
    stride=1
)
  model.aux_classifier = None
  '''
  model=custom_net.Net()


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
    validation(model,dataset,epoch=1,folder='data/test')


def validation(model,dataset,epoch,folder=None):
    model.eval()
    if folder!= None:
      if not os.path.exists(folder):
          os.mkdir(folder)
      cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
      cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for i in range(len(dataset)):      
      inputs, target,_ = dataset[i]

      img = (inputs.permute(1, 2, 0).data.numpy()*255).squeeze().astype(np.int32)
      mask = target.numpy().astype(np.uint8)
      inputs = Variable(inputs.cuda())

      outputs = model(inputs.unsqueeze(0))#['out']
      torch.cuda.synchronize()
      #time.sleep(100)
      start = time.time()

      _, pred = torch.max(outputs, 1)



      pred = pred.data.cpu()



      pred = pred.numpy().squeeze().astype(np.uint8)



      imname = dataset.masks[i].split('/')[-1]

      redImg = np.zeros(img.shape, img.dtype)
      redImg[:,:] = (0, 0, 255)
      redMask = cv2.bitwise_and(redImg, redImg, mask=pred)
      #mask_pred.save(os.path.join(folder, imname))

      img = img.copy().astype(np.float32)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB ).astype(np.int32)
      cv2.addWeighted(redMask, 0.5, img, 1, 0, img)
      cv2.imwrite(os.path.join(folder, imname),img)




      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
    logwriter.writerow([epoch+1,iou[0]*100,iou[1]*100,iou.mean()*100])  
    model.train()

if __name__ == "__main__":
  main()
