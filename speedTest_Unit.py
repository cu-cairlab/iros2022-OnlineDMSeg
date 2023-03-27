import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import cv2
import time

from torch.nn import functional as F

import csv

from torch2trt import torch2trt




from ImageMask2Tiles_uniform import processImg
from ImageMask2Tiles_uniform import assemblyImg





fp = 'reference.jpg'
img_dir = '/media/Data/row001_auto/color/cam0_images/'
img_list = os.listdir(img_dir)

input_img = np.asarray(Image.open(fp).convert('RGB'))
input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
start_time = time.time()
i = 0
for fn in img_list:
    i += 1
    fp = img_dir + '/' + fn
    input_img = np.asarray(Image.open(fp).convert('RGB'))
    input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
    tiles = processImg(input_img)
print((time.time()-start_time)/(i))









