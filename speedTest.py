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



def speedTest(args,img_dir,model,is_seperate_classifier,classifier=None):


    data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    fp = 'reference.jpg'
    #img_list = os.listdir(img_dir)

    input_img = np.asarray(Image.open(fp).convert('RGB'))
    input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
    tiles = processImg(input_img)


    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    #for fn in img_list:
    for i in range(200):
        '''
        fp = img_dir + '/' + fn
        input_img = np.asarray(Image.open(fp).convert('RGB'))
        input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
        tiles = processImg(input_img)
        '''
        output_results = {}
        for idx,tile in enumerate(tiles):
            #y = idx%3
            #x = int(idx/3)
            #print((x,y))
            input_data = data_transforms(tile.image).unsqueeze_(0).cuda().half()
            if is_seperate_classifier:
                size = (input_data.shape[2], input_data.shape[3])
            #start = time.time()
            with torch.no_grad():
                if args.net == 'Attanet':
                    outputs = model(input_data)[0]
                else:
                    outputs = model(input_data)
                if is_seperate_classifier:
                    outputs = classifier(outputs)
                    outputs = nn.Upsample(size, mode='bilinear', align_corners=False)(outputs)
            _, pred = torch.max(outputs, 1)
   
            #output_results[(x,y)] =  pred.data.cpu().numpy().squeeze().astype(np.uint8)

        #torch.cuda.synchronize()
        #assembledImg = assemblyImg(output_results)



    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time = elapsed_time_ms/1000.
    '''
    print(("total: %d", len(img_list)))
    print(elapsed_time)
    print(elapsed_time/len(img_list))
    '''
    print("total: %d", i)
    print(elapsed_time)
    print(elapsed_time/(i+1))





