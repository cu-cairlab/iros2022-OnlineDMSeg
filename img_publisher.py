#!/usr/bin/env python

import rospy

from std_msgs.msg import MultiArrayDimension,Int8MultiArray


from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import argparse
import message_filters

import io
import PIL.Image


class ImagePublisher(object):
    def __init__(self,imgDir,outputDir):
        self.br = CvBridge()
        # Node cycle rate (in Hz).

        # Publishers
        self.imgFiles = os.listdir(imgDir)
        self.imgFolder = imgDir
        self.outputDir = outputDir
        self.processDict = {}
        #self.imgFiles = [imgFolder+ '/' + fn for fn in self.imgFiles]


        rospy.init_node('talker', anonymous=True)

        image_sub = rospy.Subscriber("img_output", Int8MultiArray,self.callback)




        #self.pub1 = rospy.Publisher('img_input', Image,queue_size=10,tcp_nodelay=True)
        self.pub1 = rospy.Publisher('img_input', Int8MultiArray,queue_size=10)
        self.talker()




    def callback(self,data):
        rospy.loginfo('Image received...')
        image = self.br.imgmsg_to_cv2(data0)
        img_id = int(data1.fluid_pressure)
        img_fn = self.processDict[img_id]
        cv2.imwrite(self.outputDir+'/'+img_fn[:-4]+'.png', image)
        del self.processDict[img_id]
        print self.processDict



    def talker(self):
        rate = rospy.Rate(4) # 10hz
        img_id = 0
        dtype = np.dtype('int8')
        for img_fn in self.imgFiles:
            if rospy.is_shutdown():
                exit()
            fp = self.imgFolder+'/' +img_fn
            #image = cv2.imread(fp)
            #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
            #print image.nbytes
            #img_msg = self.br.cv2_to_imgmsg(image)
            #img_msg.header.stamp = rospy.Time.now()


 
            image =  open(fp, "rb")
            numpy_data = np.fromfile(image,dtype)
            print numpy_data.nbytes
            byte_array = numpy_data.tolist()
            #print bytes(numpy_data.tolist())
            img_msg = Int8MultiArray()
            img_msg.data = byte_array
            dims = [MultiArrayDimension()]
            dims[0].label = str(img_id)
            img_msg.layout.dim=dims

            self.pub1.publish(img_msg)


            img_id += 1
            rate.sleep()
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('imgDir', type=str, default='',
                    help='path to data')

    parser.add_argument('outputDir', type=str, default='',
                    help='path to data')

    args = parser.parse_args()
    ImagePublisher(args.imgDir,args.outputDir)

