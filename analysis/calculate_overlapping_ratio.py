
import os
import argparse
import cv2
import numpy as np

def getImageRepetitive(fp):
    repetitiveMask = cv2.imread(fp,0)
    repetitiveMask = (repetitiveMask!=0)
    return repetitiveMask

if __name__=="__main__":
    print "check"
    row_folders = ['row001_auto_out','row002_auto_out','row002_auto_out']

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_root', type=str, default="/home/leo/Nvidia_SS/semantic-segmentation2/logs/eval_DM/dexterous-bee_2021.04.21_13.32/best_images/")
    parser.add_argument('repetitive_folder', type=str, default="/home/leo/Nvidia_SS/semantic-segmentation2/logs/eval_DM/dexterous-bee_2021.04.21_13.32/best_images/")
    args = parser.parse_args()

    data_root = args.data_root
    repetitive_folder = args.repetitive_folder
    
    overlapping = 0
    total = 0
    for row_folder in row_folders:
        repetitive_mask_dir = data_root + '/' + row_folder + '/' + repetitive_folder + '/'
        for fn in os.listdir(repetitive_mask_dir):
            repetitiveMask = getImageRepetitive(repetitive_mask_dir+'/'+fn)
            total += repetitiveMask.size
            overlapping += np.sum(repetitiveMask == 1)
    print overlapping*1./total







    
    
