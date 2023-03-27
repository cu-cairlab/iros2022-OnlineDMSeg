import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import cv2
import time
import argparse
from scipy.io import loadmat
from torch.nn import functional as F

import csv

from torch2trt import torch2trt

from torchvision.models.segmentation.deeplabv3 import DeepLabHead as classifier

from utils import AverageMeter, inter_and_union

from DM import DMData



from ImageMask2Tiles_uniform import processImg
from ImageMask2Tiles_uniform import assemblyImg





def accuracyTest(args,dataset,model_trt,is_seperate_classifier,classifier,folder):

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for i in range(len(dataset)):
        inputs, target,index = dataset[i]
        mask = target.numpy().astype(np.uint8)
        input_data = Variable(inputs.cuda()).unsqueeze(0).half()


   
        size = (input_data.shape[2], input_data.shape[3])

        with torch.no_grad():
  
            if args.net == 'Attanet':
                outputs = model_trt(input_data)[0]
            else:
                outputs = model_trt(input_data)

            if is_seperate_classifier:
                outputs = classifier(outputs)
                outputs = nn.Upsample(size, mode='bilinear', align_corners=False)(outputs)
        


        _, pred = torch.max(outputs, 1)

        pred = pred.data.cpu()



        pred = pred.numpy().squeeze().astype(np.uint8)


        #'''
        imname = dataset.masks[i].split('/')[-1]
        img = cv2.imread(dataset.images[index]).astype(np.int32)
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
 
        img = img.copy().astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB ).astype(np.int32)
        cv2.addWeighted(redMask, 0.25, img, 1, 0, img)
        cv2.addWeighted(blueImg, 0.25, img, 1, 0, img)
        cv2.imwrite(os.path.join(folder+'/trt/', imname),img)
        #'''



        inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
        inter_meter.update(inter)
        union_meter.update(union)





    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))













