import argparse
import os
import shutil
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import csv

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=True,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--dataset', type=str, default='DM',
                    help='pascal or cityscapes')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--crop_size', type=int, default=512,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
args = parser.parse_args()


def main():
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'DM':
    dataset = DMData('/media/data/DM4/',
        train=True, crop_size=args.crop_size)

    valDataset = DMData('/media/data/DM4/',
        train=False, crop_size=args.crop_size)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

  nb_classes = 3
  '''
  model = segmentation.deeplabv3_resnet101(pretrained=True,num_classes=21)
  model.classifier[4]=nn.Conv2d(
    in_channels=256,
    out_channels=nb_classes,
    kernel_size=1,
    stride=1
)
  model.aux_classifier = None
  '''
  model=custom_net.Net()


  if args.train:

    best_score = 0
    best_model_name = '-1'
    last_model_name = '-1'
    now = datetime.now()
    logdir = '../ignored/'+args.exp+'/'+now.strftime("%d%m%Y_%H%M%S")+'/'
    if os.path.exists(logdir):
        Question = input("log folder exists. Delete? (yes|no)")
        if Question == ("yes"):
            shutil.rmtree(logdir)
            print ("log deleted")
        elif Question == ("no"):
            print ("log kept")
    tensorboardWriter = SummaryWriter(logdir)
    '''
    trainLogFile = open('trainLog.csv','w')
    trainLogwritter = csv.writer(trainLogFile)

    valLogFile = open('valLog.csv','w')
    valLogwritter = csv.writer(valLogFile)

    valTrainLogFile = open('valTrainLog.csv','w')
    valTrainLogwritter = csv.writer(valTrainLogFile)
    '''

    weights = [0., 1.0,0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255,weight=class_weights)
    model = nn.DataParallel(model).cuda()
    model.train()


    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False

    optimizer = optim.Adam(model.parameters(),lr=args.base_lr, weight_decay=0.004)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.train,
        pin_memory=True, num_workers=args.workers,drop_last=True)
    max_iter = args.epochs * len(dataset_loader)
    #losses = AverageMeter()
    start_epoch = 0

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))


    THRESHOLD = 5
    for epoch in range(start_epoch, args.epochs):
      losses = AverageMeter()

      is_bestScore = False

      if epoch > THRESHOLD:
          weights = [0.1, 1.0,0]
          class_weights = torch.FloatTensor(weights).cuda()
          criterion = nn.CrossEntropyLoss(ignore_index=255,weight=class_weights)
      for i, (inputs, target,_) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9

        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        outputs = model(inputs)
        #print(outputs.size())
        #print(target.size())
        #print(torch.unique(target))
        loss = criterion(outputs, target)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
          pdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

      tensorboardWriter.add_scalar('Loss/train', losses.avg,epoch)
      #after iterating dataset:
      #trainLogwritter.writerow([epoch+1,losses.avg]) 
      

      if epoch % 5 == 4:
        with torch.no_grad():
          score = validation(model,valDataset,epoch,folder=logdir+'/val/')
          score_train = validation(model,dataset,epoch)

        tensorboardWriter.add_scalar('iou/train', score_train,epoch)

        tensorboardWriter.add_scalar('Loss/val', score,epoch)

        if score > best_score and score>0.5:
            best_score = score

            if best_model_name != '-1':
                os.remove(best_model_name)
            best_model_name= logdir+"best_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_model_name)
 


 
      if epoch % 50 == 49:
        #validation(model,valDataset,valLogwritter,epoch,folder='data/val')
        #validation(model,dataset,valTrainLogwritter,epoch)
        if last_model_name != '-1':
            os.remove(last_model_name)
        last_model_name = logdir+"epoch%d" %(epoch + 1)
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, last_model_name)
  

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
      inputs = Variable(inputs.cuda())
      #print(inputs.size())
      outputs = model(inputs.unsqueeze(0))
      _, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
      mask = target.numpy().astype(np.uint8)
      imname = dataset.masks[i].split('/')[-1]
      if folder!= None:
        mask_pred = Image.fromarray(pred)
        mask_pred.putpalette(cmap)
        mask_pred.save(os.path.join(folder, imname))

      print('eval: {0}/{1}'.format(i + 1, len(dataset)))

      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    #print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
    #logwriter.writerow([epoch+1,iou[0]*100,iou[1]*100,iou.mean()*100])  
    model.train()
    return iou.mean()

if __name__ == "__main__":
  main()
