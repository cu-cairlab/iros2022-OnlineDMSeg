import argparse
import os
import shutil
import numpy as np
import cv2
import time
import pickle
from Xavier_SpeedTest_folder_function import imageProcessor
import torch

import os




def processResult(imgProcessor,img_folder,row_segmentation,output_dir,mask_output_dir,effectiveImgList=None):

    isOutput = mask_output_dir != 'None'

    keyed_dict = {}
    keyed_dict_detailed = {}
    keyed_dict_Image_detailed = {}

    if isOutput:
        if os.path.exists(mask_output_dir):
            shutil.rmtree(mask_output_dir)
        os.mkdir(mask_output_dir)

    start_time = time.time()
    img_count = 0
    if True:
        for frame in os.listdir(img_folder):
            frame = frame[:-4]
            print(frame)

            
            if True:
                #print frame
                fn = frame+'.png'
                fp = img_folder+'/'+fn
                print(fn)
                img_count = img_count+1
                input_img = cv2.imread(fp)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                #input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE) 
                torch.cuda.synchronize()
                start_time2 = time.time()
                infectedArea,leaveArea,assembledImg = imgProcessor.inference(input_img,dump=isOutput)
                torch.cuda.synchronize()
                time_used2 = time.time()-start_time2
                if isOutput:
                    print('saved')
                    imname = frame+'_prediction.png'
                    cv2.imwrite(mask_output_dir+'/'+imname,assembledImg,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(time_used2)

    time_used = time.time()-start_time

    return time_used,img_count



parser = argparse.ArgumentParser()

#parser.add_argument('--weight', type=str, default=None,
#                    help='path to checkpoint to resume from')
parser.add_argument('--dataPath', type=str, default='',
                    help='path to data')


parser.add_argument('--row_segmentation', type=str, default='None',
                    help='path to data')

parser.add_argument('--output', type=str, default='None',
                    help='path to data')
parser.add_argument('--img_dump_dir', type=str, default='None',
                    help='path to data')

parser.add_argument('--downsample', type=int, default=1,
                    help='path to data')



args = parser.parse_args()


img_folder = args.dataPath
output_dir = args.output
mask_output_dir = args.img_dump_dir
downsample = args.downsample

os.makedirs(output_dir, exist_ok=True )


#downsample
if downsample == 1:
    effectiveImgList=None
else:
    effectiveImgList = os.listdir(img_folder)
    effectiveImgList.sort()
    effectiveImgList = effectiveImgList[::downsample]
#load row segmentation plan
'''
if args.row_segmentation[-4:] == '.csv':
    row_segmentation = csv2dict.csv2dict(args.row_segmentation,args.image_path)
elif args.row_segmentation[-4:] == '.pkl':
    row_segmentation = pickle.load( open( args.row_segmentation, "rb" ) )
else:
    print(args.row_segmentation)
    print("unsupported row segmentation")
    assert False
'''
row_segmentation = None
imgProcessor = imageProcessor('/media/nvidia/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1.trt','reference.png')

start_time = time.time()
time_used,img_count = processResult(imgProcessor,img_folder,row_segmentation,output_dir,mask_output_dir,effectiveImgList)
print("Total time:")
print(time.time()-start_time)
print("per image time:")
print(time_used*1./img_count)

