#!/usr/bin/env python3
import rospy

from std_msgs.msg import MultiArrayDimension,Int8MultiArray
import os
import shutil
import cv2
from cv_bridge import CvBridge
import ros_numpy
import numpy as np
import PIL.Image
import io

from torch2trt import torch2trt
from torch2trt import TRTModule

import torch
from torchvision import transforms


from ImageMask2Tiles_uniform import processImg
from ImageMask2Tiles_uniform import assemblyImg


from DeepLabv3_modified_addBN import custom_net


class imageProcessor(object):

    def __init__(self):
        rospy.init_node('listener', anonymous=True)

        self.image_sub = rospy.Subscriber("img_input", Int8MultiArray,self.callback)
        self.pub1 = rospy.Publisher('img_input', Int8MultiArray,queue_size=10,tcp_nodelay=True)
        self.listener()


    def callback(self,data):
        arr = np.array(data.data,np.dtype('int8'))
        image = np.array(PIL.Image.open(io.BytesIO(arr)))
        img_id = data.layout.dim[0].label
        cv2.imwrite('/home/cairlab/cache/'+str(img_id)+'.jpg', image)



 
    def listener(self):


        while not rospy.is_shutdown():
            rospy.spin()

            #self.loop_rate.sleep()



if __name__ == '__main__':

    if os.path.exists('/home/cairlab/cache/'):
        shutil.rmtree('/home/cairlab/cache/')
    os.mkdir('/home/cairlab/cache/')


    imageProcessor()

