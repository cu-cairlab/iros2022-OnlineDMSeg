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

parser = argparse.ArgumentParser()


parser.add_argument('--reference_path', type=str, default='',
                    help='path to data')

parser.add_argument('--input_path', type=str, default='',
                    help='path to data')

args = parser.parse_args()


input_path=  args.input_path

input_list = os.listdir(input_path)

for i in range(len(input_list)):
    input_fn = input_list[i]
    input_fp = input_path+'/'+input_fn
    reference_fp = reference_path+input_fn

    inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
    inter_meter.update(inter)
    union_meter.update(union)


print(("test %d" %i))
print(time.time()-start)
print((time.time()-start)/len(dataset))


iou = inter_meter.sum / (union_meter.sum + 1e-10)
for i, val in enumerate(iou):
  print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))





