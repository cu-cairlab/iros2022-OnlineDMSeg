import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import random

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

parser.add_argument('--net', type=str, default='DeepLabv3_modified_addBN',
                    help='path to data')


parser.add_argument('--num_classes', type=int, default=3,
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
parser.add_argument('--fix_kernel_size', type=bool, default=True)



args = parser.parse_args()



print("switches:")
print("use_concat:",args.use_concat)
print("multiLoss:",args.multiLoss)
print("fix_kernel_size:",args.fix_kernel_size)
print("freeze_bn:",args.freeze_bn)




def main():


  nb_classes = args.num_classes

  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'DM':
    dataset = DMData('/data/DM4/',
        train=True, crop_size=args.crop_size)
    
    valDataset = DMData('/data/DM4/',
        train=False, crop_size=args.crop_size)
    nb_classes = len(dataset.CLASSES)+1
    print("nb_class changed to: %d" %nb_classes)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))



  #get net


  weights_idx = 0

  #thresholds = [0,25,1000]
  #weights = [[0, 1.0,0.,0],[0.01, 1.0,0.01,0],[0.2, 1.0,0.2,0]]
  thresholds = [0,50]
  weights = [[0, 1.0,0.,0],[0.2, 1.0,0.2,0]]
  '''
  elif args.net == 'FCN':

      from FCN import fcn as custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone)
  '''
  if args.net == 'DeepLabv3_modified_addBN':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,
             feature_map_size_layer=args.feature_map_size_layer,feature_accumumation_start=args.feature_accumumation_start,
             pretrained=True,concat = args.use_concat,name_classifier=args.name_classifier,fix_kernel_size = args.fix_kernel_size)
      is_seperate_classifier = False


  elif args.net == 'HED':

      from DeepLabv3_modified_addBN import custom_net
      model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,feature_map_size_layer=args.feature_map_size_layer,
                             feature_accumumation_start=args.feature_accumumation_start,pretrained=False,original=True,multiLoss=args.multiLoss,name_classifier=args.name_classifier)
      if args.multiLoss:
          if args.feature_channel > 1:
              aux_head1 = FCNHead(64,nb_classes).cuda()
              aux_head2 = FCNHead(64,nb_classes).cuda()
              aux_head3 = FCNHead(64,nb_classes).cuda()
              aux_head4 = FCNHead(64,nb_classes).cuda()
          else:
              aux_head1 = FCNHead(1,nb_classes).cuda()
              aux_head2 = FCNHead(1,nb_classes).cuda()
              aux_head3 = FCNHead(1,nb_classes).cuda()
              aux_head4 = FCNHead(1,nb_classes).cuda()

      is_seperate_classifier = False

  elif args.net == 'Deeplabv3':
      from deeplabv3_model import deeplabv3
      model = deeplabv3.DeepLabV3(nb_classes,backbone=args.backbone,name_classifier=args.name_classifier)

  elif args.net == 'SFSegNet':

      #from network.sfnet_resnet import DeepR18_SF_deeply
      #model = DeepR18_SF_deeply(nb_classes, criterion=None)
      if args.backbone == 'resnet':
          from sfnet.sfnet_resnet import DeepR18_SF_deeply
          model = DeepR18_SF_deeply(nb_classes, criterion=None)
      else:
          from sfnet.sfnet_dfnet import AlignedDFnetv2_FPNDSN
          model = AlignedDFnetv2_FPNDSN(nb_classes, criterion=None)

  elif args.net == 'Attanet':

      #from network.sfnet_resnet import DeepR18_SF_deeply
      #model = DeepR18_SF_deeply(nb_classes, criterion=None)
      from Attanet import AttaNet
      model = AttaNet.AttaNet(nb_classes)


  else:
      print(args.net)
      print("net not defined")


  if args.train:

    best_score = 0
    best_dm_score = 0
    best_model_name = '-1'
    best_dm_model_name = '-1'
    last_model_name = '-1'
    now = datetime.now()
    logdir = './ignored/'+args.exp+'/'+now.strftime("%d%m%Y_%H%M%S")+'/'
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

    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255,weight=class_weights)

    if args.net == 'Attanet':
        from loss import OhemCELoss

        score_thres = 0.7
        n_min = args.crop_size*args.crop_size//2

        criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
        criteria_aux1 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
        criteria_aux2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)




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



    for epoch in range(start_epoch, args.epochs):
      losses = AverageMeter()

      is_bestScore = False

      if weights_idx < len(thresholds) and epoch >= thresholds[weights_idx]:
          weight = weights[weights_idx]
          class_weight = torch.FloatTensor(weight).cuda()
          criterion = nn.CrossEntropyLoss(ignore_index=255,weight=class_weight)
          weights_idx += 1
      for i, (inputs, target,_) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9

        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        if args.net != 'Attanet':
            outputs = model(inputs)
        if args.net == 'DeepLabv3_original':
          outputs = outputs["out"]
        #print(outputs.size())
        #print(target.size())
        #print(torch.unique(target))

        if args.net == 'Attanet':
            out,out16,out32 = model(inputs)
            lossp = criteria_p(out, target)
            loss1 = criteria_aux1(out16, target)
            loss2 = criteria_aux2(out32, target)
            loss = lossp + loss1 + loss2

        if args.net == 'HED' and args.multiLoss:
            #target_aux = target
            out,out1,out2,out3,out4 = model(inputs)
            #print(out.size())
            #print(out1.size())
            #print(out2.size())
            #print(out3.size())
            #print(out4.size())
            out1 = aux_head1(out1)
            out2 = aux_head2(out2)
            out3 = aux_head3(out3)
            out4 = aux_head4(out4)
            out1 = nn.Upsample((target.size()[1], target.size()[2]), mode='bilinear', align_corners=False)(out1)
            out2 = nn.Upsample((target.size()[1], target.size()[2]), mode='bilinear', align_corners=False)(out2)
            out3 = nn.Upsample((target.size()[1], target.size()[2]), mode='bilinear', align_corners=False)(out3)
            out4 = nn.Upsample((target.size()[1], target.size()[2]), mode='bilinear', align_corners=False)(out4)
 
            loss1 = criterion(out, target)
            loss2 = criterion(out1, target)
            loss3 = criterion(out2, target)
            loss4 = criterion(out3, target)
            loss5 = criterion(out4, target)
            loss = loss1+loss2+loss3+loss4+loss5


        else:
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
          score_train = validation(model,dataset,epoch,random_select=40)

        tensorboardWriter.add_scalar('iou/train', score_train.mean(),epoch)

        tensorboardWriter.add_scalar('iou/val', score.mean(),epoch)
        for i, val in enumerate(score):
          #print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
          tensorboardWriter.add_scalar('iou/val/' + dataset.CLASSES[i], val,epoch)


        for i, val in enumerate(score_train):
          #print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
          tensorboardWriter.add_scalar('iou/train/' + dataset.CLASSES[i], val,epoch)



        dm_score = score[1]
        dm_train_score = score_train[1]

        score = score.mean()
        score_train = score_train.mean()



        if score > best_score and score>0.5 and not args.multiLoss:
            best_score = score

            if best_model_name != '-1':
                os.remove(best_model_name)
            best_model_name= logdir+"best_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_model_name)


        if dm_score > best_dm_score and dm_score>0.35 and not args.multiLoss:
            best_dm_score = dm_score

            if best_dm_model_name != '-1':
                os.remove(best_dm_model_name)
            best_dm_model_name= logdir+"best_dm_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_dm_model_name)


        if score > best_score and score>0.5 and args.multiLoss:
            best_score = score

            if best_model_name != '-1':
                os.remove(best_model_name)
            best_model_name= logdir+"best_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'aux_head1':aux_head1.state_dict(),
              'aux_head2':aux_head2.state_dict(),
              'aux_head3':aux_head3.state_dict(),
              'aux_head4':aux_head4.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_model_name)


        if dm_score > best_dm_score and dm_score>0.35 and args.multiLoss:
            best_dm_score = dm_score

            if best_dm_model_name != '-1':
                os.remove(best_dm_model_name)
            best_dm_model_name= logdir+"best_dm_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'aux_head1':aux_head1.state_dict(),
              'aux_head2':aux_head2.state_dict(),
              'aux_head3':aux_head3.state_dict(),
              'aux_head4':aux_head4.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_dm_model_name)



 


 
      if epoch % 50 == 49 and not args.multiLoss:
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
  

      if epoch % 50 == 49 and args.multiLoss:
        #validation(model,valDataset,valLogwritter,epoch,folder='data/val')
        #validation(model,dataset,valTrainLogwritter,epoch)
        if last_model_name != '-1':
            os.remove(last_model_name)
        last_model_name = logdir+"epoch%d" %(epoch + 1)
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'aux_head1':aux_head1.state_dict(),
          'aux_head2':aux_head2.state_dict(),
          'aux_head3':aux_head3.state_dict(),
          'aux_head4':aux_head4.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, last_model_name)

def validation(model,dataset,epoch,folder=None, random_select=0):
    model.eval()
    if folder!= None:
      if not os.path.exists(folder):
          os.mkdir(folder)
      cmap = loadmat('pascal_seg_colormap.mat')['colormap']
      cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    if random_select == 0 or random_select > len(dataset):
        iterationList = range(len(dataset))
    else:
        iterationList = random.sample(range(0, len(dataset)), random_select)



    for num,i in enumerate(iterationList):
      inputs, target,_ = dataset[i]
      inputs = Variable(inputs.cuda())
      #print(inputs.size())
      outputs = model(inputs.unsqueeze(0))

      if args.net == 'DeepLabv3_original':
          outputs = outputs["out"]

      if args.net == 'Attanet' or args.multiLoss:
          outputs = outputs[0]


      _, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
      mask = target.numpy().astype(np.uint8)
      imname = dataset.masks[i].split('/')[-1]
      if folder!= None:
        mask_pred = Image.fromarray(pred)
        mask_pred.putpalette(cmap)
        mask_pred.save(os.path.join(folder, imname))

      print('eval: {0}/{1}'.format(num + 1, len(dataset)))

      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    #print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
    #logwriter.writerow([epoch+1,iou[0]*100,iou[1]*100,iou.mean()*100])  
    model.train()
    return iou#.mean()

if __name__ == "__main__":
  main()
