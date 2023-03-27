import os
import time

import cv2
import numpy as np
import PIL.Image
import io

from torch2trt import torch2trt
from torch2trt import TRTModule

import torch
from torchvision import transforms




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


def load_model(args):

    nb_classes = args.num_classes
    backbone = args.backbone
    if True:

        if args.net == 'DeepLabv3_modified_addBN':

            from DeepLabv3_modified_addBN import custom_net
            model=custom_net.Net(nb_classes,backbone=args.backbone,feature_channel=args.feature_channel,feature_map_size_layer=args.feature_map_size_layer,feature_accumumation_start=args.feature_accumumation_start,pretrained=False)
            is_seperate_classifier = False

        elif args.net == 'Resnet50_New':

            from DeepLabv3_modified_addBN import resnet_custom as resnet
            if backbone=="resnet50":
                model=resnet.resnet50(replace_stride_with_dilation = [False, False, False],feature_channel=args.feature_channel)
            elif backbone=="resnet34":
                model=resnet.resnet34(replace_stride_with_dilation = [False, False, False],feature_channel=args.feature_channel)
                classifier = classifier(args.feature_channel,nb_classes)
      
            is_seperate_classifier = True

 

        elif args.net == 'SFSegNet':

            from network.sfnet_resnet import DeepR18_SF_deeply
            model = DeepR18_SF_deeply(nb_classes, criterion=None)
            is_seperate_classifier = False
        elif args.net == 'Attanet':

            #from network.sfnet_resnet import DeepR18_SF_deeply
            #model = DeepR18_SF_deeply(nb_classes, criterion=None)
            from Attanet import AttaNet
            model = AttaNet.AttaNet(nb_classes,pretrained=False)
            is_seperate_classifier = False

        else:
            print(args.net)
            print("net not defined")




        if os.path.isfile(args.weight):
            print('=> loading checkpoint {0}'.format(args.weight))
            if is_seperate_classifier:
                model,classifier,epoch = loadWeight(model,classifier,args.weight)
            else:
                checkpoint = torch.load(args.weight)
                epoch = checkpoint['epoch']
                state_dict_model = {}
                for key in checkpoint['state_dict'].keys():
                    key_split = key.split('.')

                    newkey = ".".join(key_split[1:])
                    #print(newkey)
                    state_dict_model[newkey] = checkpoint['state_dict'][key]

                model.load_state_dict(state_dict_model)

            print('=> loaded checkpoint {0} (epoch {1})'.format(
              args.weight, epoch))

        else:
            print('=> no checkpoint found at {0}'.format(args.weight))
    return model



class imageProcessor(object):

    def __init__(self,modelDir,runUpImgDir,ImageMask2Tiles_instance,args=None):
        self.ImageMask2Tiles = ImageMask2Tiles_instance
        if args != None:


            model = load_model(args).cuda().eval().half()

            print("+++++++ trt +++++++++++")
            bs = 3
            w = 1125
            h = 1352
            input_sample = torch.rand(bs,3, w, h).cuda().half()
            model_trt = torch2trt(model, [input_sample],max_batch_size=bs,fp16_mode=True)
            torch.save(model_trt.state_dict(), 'trtModel.pth')
            start = time.time()
            model_trt = TRTModule()

            model_trt.load_state_dict(torch.load('trtModel.pth'))
            print(time.time()-start)
            print("------loaded--------")
            
            self.model = model_trt


        else:
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(modelDir))




        self.data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        #run up
        print('run up')
        print('should take ~30s')
        image = cv2.imread(runUpImgDir)
        #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        for i in range(50):
            self.inference(image)
        print('run up completed')       
        print("ready") 

        del image


    def inference(self,input_img,dump=False):


        #input_img = cv2.imread(fp)
        output_results = {}
        infected_area = 0
        canopy_area = 0
        
        tiles = self.ImageMask2Tiles.processImg(input_img)
        tiles_keys = tiles.keys()
        start = time.time()
        for key in tiles_keys:
            tile = tiles[key]
            #print((x,y))
            input_data = tile #self.data_transforms(tile).unsqueeze_(0).cuda().half()
            with torch.no_grad():
                outputs = self.model(input_data)#[0]
            _, pred = torch.max(outputs, 1)
            #mask_output =  pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask_output = pred.data.squeeze()
            leaf = int(torch.sum(mask_output != 0))
            infection = int(torch.sum(mask_output == 1))
            if leaf > 1E-6:
                infected_area = infected_area + infection
                canopy_area = canopy_area + leaf
            else:
                infected_area = infected_area + 0
                canopy_area = canopy_area + 0

            if dump: 
                output_results[key] = mask_output.cpu().numpy().astype(np.uint8)
        print("model running: ", time.time()-start)
        #print([infected_area,canopy_area])


        if dump:
            assembledImg = self.ImageMask2Tiles.assemblyImg(output_results).astype('uint8')


            
            redImg = np.zeros(input_img.shape, input_img.dtype)
            redImg[:,:] = (0, 0, 255)
            leafMask = np.copy(assembledImg)
            leafMask[leafMask==1] = 0
            leafMask[leafMask==2] = 1

            print(redImg.shape)
            print(leafMask.shape)

            redMask = cv2.bitwise_and(redImg, redImg, mask=leafMask)
            blueImg = np.zeros(input_img.shape, input_img.dtype)
            blueImg[:,:] = (0, 255, 255)
            DMMask = np.copy(assembledImg)
            DMMask[DMMask==2] = 0
            blueImg = cv2.bitwise_and(blueImg, blueImg, mask=DMMask)
 
            cv2.addWeighted(blueImg, 0.5, redMask, 0.5, 0, blueImg)

            assembledImg = blueImg




            return infected_area,canopy_area,assembledImg
        else:
            return infected_area,canopy_area,None



if __name__ == '__main__':
    imageProcessor('ignored/trtModel.pth','reference.jpg')





