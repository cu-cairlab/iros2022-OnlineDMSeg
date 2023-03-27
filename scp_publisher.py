import cv2
import os
import numpy as np
import argparse
import message_filters

import io
import PIL.Image
import time

from subprocess import call

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('imgDir', type=str, default='',
                    help='path to data')

    parser.add_argument('outputDir', type=str, default='',
                    help='path to data')

    args = parser.parse_args()
    '''


    rootDir = '/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5/row001_auto_out/rectified0/'
    img_fns = os.listdir(rootDir)
    img_fns.sort()
    for img_fn in img_fns:
        print img_fn
        start = time.time()
        img_fp = rootDir+img_fn
        cmd = "sshpass -p Cornell scp " + img_fp + " cairlab@192.168.0.200:/home/cairlab/cache/"
        call(cmd.split(" "))
        print time.time()-start
