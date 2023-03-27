import argparse
import os
import shutil
import numpy as np
import cv2
import time
import pickle
from Xavier_SpeedTest_folder_function import imageProcessor


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
    for seg_id in row_segmentation.keys():
        frame_sequences = row_segmentation[seg_id]
        leaveTotal = 0
        infectedTotal = 0
        max_infection_rate = 0
        for frames in frame_sequences:
            
            print(seg_id)
            print(frames)

            mid_frame_id = int(len(frames)/2)
            for frame_idx,frame in enumerate(frames):
                #print frame
                fn = frame+'.jpg'
                #if frame_idx == mid_frame_id:
                fp = img_folder+'/'+fn
                if effectiveImgList!=None and fn not in effectiveImgList:
                    continue
                print(fn)
                img_count = img_count+1
                input_img = cv2.imread(fp)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE) 
                infectedArea,leaveArea,assembledImg = imgProcessor.inference(input_img,dump=isOutput)
                infectedTotal += infectedArea
                leaveTotal += leaveArea
                if isOutput:
                    imname = frame+'_prediction.png'
                    cv2.imwrite(mask_output_dir+'/'+imname,assembledImg,[cv2.IMWRITE_PNG_COMPRESSION, 0])


        if leaveTotal>1E-6:
            keyed_dict[seg_id] = infectedTotal*1.0/leaveTotal
        else:
            keyed_dict[seg_id] = -1
        keyed_dict_detailed[seg_id] = [infectedTotal,leaveTotal]

    time_used = time.time()-start_time

    print(keyed_dict)
    f = open(output_dir + "/section_infection.pkl","wb")
    pickle.dump(keyed_dict,f,protocol=2)
    f.close()

    f = open(output_dir + "/section_infection_detailed.pkl","wb")
    pickle.dump(keyed_dict_detailed,f,protocol=2)
    f.close()

    f = open(output_dir + "/section_infection_image_detailed.pkl","wb")
    pickle.dump(keyed_dict_Image_detailed,f,protocol=2)
    f.close()
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
if args.row_segmentation[-4:] == '.csv':
    row_segmentation = csv2dict.csv2dict(args.row_segmentation,args.image_path)
elif args.row_segmentation[-4:] == '.pkl':
    row_segmentation = pickle.load( open( args.row_segmentation, "rb" ) )
else:
    print(args.row_segmentation)
    print("unsupported row segmentation")
    assert False


imgProcessor = imageProcessor('ignored/trtModel.pth','reference.jpg')

start_time = time.time()
time_used,img_count = processResult(imgProcessor,img_folder,row_segmentation,output_dir,mask_output_dir,effectiveImgList)
print("Total time:")
print(time.time()-start_time)
print("per image time:")
print(time_used*1./img_count)

