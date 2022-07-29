#!/usr/bin/env python3
import rospy

from std_msgs.msg import MultiArrayDimension,Int8MultiArray
import os
import cv2
from cv_bridge import CvBridge
import ros_numpy
import numpy as np
import PIL.Image
from torch2trt import torch2trt
from torch2trt import TRTModule

import torch
from torchvision import transforms


from ImageMask2Tiles_uniform import processImg
from ImageMask2Tiles_uniform import assemblyImg


from DeepLabv3_modified_addBN import custom_net
      
def load_weight(model,weightpath):
    print('=> loading checkpoint {0}'.format(weightpath))

    if True:
        checkpoint = torch.load(weightpath)
        epoch = checkpoint['epoch']
        state_dict_model = {}
        for key in checkpoint['state_dict'].keys():
            key_split = key.split('.')

            newkey = ".".join(key_split[1:])
            #print(newkey)
            state_dict_model[newkey] = checkpoint['state_dict'][key]

        model.load_state_dict(state_dict_model)
    '''
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    '''
    print('=> loaded checkpoint {0} (epoch {1})'.format(
      weightpath, epoch))
    return model


def is_valid_jpg(jpg_file): 

    if jpg_file.split('.')[-1].lower() == 'jpg':
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            return f.read() == '\xff\xd9'
    else:
        return True




class imageProcessor(object):

    def __init__(self):
        rospy.init_node('listener', anonymous=True)

        #self.model=custom_net.Net(4,backbone='resnet18',feature_channel=64)
        #self.model = load_weight(self.model,'ignored/best_model_epoch4625.pth')

        #self.model = self.model.cuda()
        #self.model.eval()

        #self.model.half()


        self.model = TRTModule()

        self.model.load_state_dict(torch.load('ignored/trtModel.pth'))


        #run up
        print('run up')
        print('should take ~30s')
        image = cv2.imread('reference.jpg')
        for i in range(50):
            self.inference(image)
        print('run up completed')       
        print("ready") 

        del image
        self.listener()





    def inference(self,input_img):


        #input_img = cv2.imread(fp)
        tiles = processImg(input_img)
        output_results = {}
        for idx,tile in enumerate(tiles):
            y = idx%3
            x = int(idx/3)
            #print((x,y))
            input_data = transforms.ToTensor()(tile.image).unsqueeze_(0).cuda().half()
            
        #start = time.time()
            with torch.no_grad():
                outputs = self.model(input_data)[0]
            _, pred = torch.max(outputs, 1)
   
            output_results[(x,y)] =  pred.data.cpu().numpy().squeeze().astype(np.uint8)

        assembledImg = assemblyImg(output_results)
        return assembledImg

 
    def listener(self):

        path = '/home/cairlab/cache/'
        while not rospy.is_shutdown():
            img_fns = os.listdir(path)
            if len(img_fns) > 0:
                img_fns.sort()
                img_id = img_fns[0]
                #image = cv2.imread(path+'/'+img_id)
                try:
                    image = np.array(PIL.Image.open(path+'/'+img_id))
                    mask = self.inference(image)
                except Exception as e:
                    print(img_id)
                    print(e)
                    print('continuing')
                    continue
                print('*img received*')
                print(img_id)
                print(len(img_fns))
                #cv2.imwrite('test.jpg', image)
                os.remove(path+'/'+img_id)                 
                #print(mask.nbytes)

        #rospy.spin()



if __name__ == '__main__':
    imageProcessor()

