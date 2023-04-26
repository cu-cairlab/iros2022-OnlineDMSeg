import PIL.Image
import numpy as np
import cv2
import json
import os
import argparse
import shutil
import time

class IMAGE:
    pass

class TILE:
    pass

class BBOX:
    pass


import torch

def to_tensor(img):
    img = torch.from_numpy(img).float().cuda().permute(2,0,1).unsqueeze_(0).half()/255.0
    #img = img/255.0
    return img


class ImageMask2Tiles():
    def __init__(self,imgShape,tileSize,transfer_cuda = False):
        self.transfer_cuda = transfer_cuda
        self.imgShape = imgShape
        self.tileSize = tileSize
        self.horizontal_tile_number = int(imgShape[1]/tileSize[1])
        self.vertical_tile_number = int(imgShape[0]/tileSize[0])
        self.new_horizontal = self.horizontal_tile_number*tileSize[1]
        self.new_vertical = self.vertical_tile_number*tileSize[0]

        horizontal_extra = imgShape[1]-self.new_horizontal
        vertical_extra = imgShape[0]-self.new_vertical
        print((self.horizontal_tile_number,self.vertical_tile_number))

        self.horizontal_offset = int(horizontal_extra/2)
        self.vertical_offset = int(vertical_extra/2)
        self.horizontal_end = self.horizontal_offset+self.new_horizontal
        self.vertical_end = self.vertical_offset+self.new_vertical
        if len(imgShape) == 3:
            self.new_imgShape = (self.new_vertical,self.new_horizontal,imgShape[2])
        else:
            self.new_imgShape = (self.new_vertical,self.new_horizontal)
            
        if self.transfer_cuda:
            from torchvision import transforms
            #self.data_transforms = transforms.Compose([
            #  transforms.ToTensor(),
            #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #])
            self.data_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def assemblyImg(self,result_tiles):
        max_x = self.horizontal_tile_number
        max_y = self.vertical_tile_number
        originalShape = self.new_imgShape
        assembledImg = np.zeros(originalShape)
        x_pointer = 0
        y_pointer = 0
    
        print(result_tiles.keys())
        for x in range(max_x):
            y_pointer = 0
            for y in range(max_y):
                #tile = cv2.imread(imgDir+imgFileDict[(x,y)])
                tile = result_tiles[(x,y)]
                #print np.unique(tile)
                h_t,w_t = tile.shape[0],tile.shape[1]
                assembledImg[y_pointer:y_pointer+h_t,x_pointer:x_pointer+w_t] = tile
                #print (x_pointer,y_pointer)
                y_pointer = y_pointer+h_t
            x_pointer = x_pointer+w_t

        assert (x_pointer - originalShape[1]) <=1
        assert (y_pointer - originalShape[0]) <=1

        return assembledImg



            
    

    def getTiles(self,im):
        if self.transfer_cuda:
            #im = self.data_transforms(im).unsqueeze_(0).cuda().half()
            
            im = to_tensor(im)
            im = self.data_normalize(im)
            #im = im.unsqueeze_(0).half()
            
        h,w = (self.new_imgShape[0],self.new_imgShape[1])

        tiles = {}
        last_x = 0
        last_y = 0

        M = self.tileSize[1]
        N = self.tileSize[0]
        idx_x = 0
        for x in range(0,w,M):
            last_x = x
            idx_y = 0
            for y in range(0,h,N):
                if self.transfer_cuda:
                    tiles[(idx_x,idx_y)] = im[:,:,y:y+N,x:x+M]
                else:
                    tiles[(idx_x,idx_y)] = im[y:y+N,x:x+M]
                last_y = y
                idx_y += 1
                #print((x,y))
                #print((idx_x,idx_y))
                
            idx_x += 1

        return tiles
   
    def processImg(self,image):




        h,w,c = image.shape
        if not (self.imgShape == image.shape):
            print("input shape:")
            print(image.shape)
            print("expected:")
            print(self.imgShape)
            assert (self.imgShape == image.shape)
        croped_image = image[self.vertical_offset:self.vertical_end, self.horizontal_offset:self.horizontal_end]
        tiles = self.getTiles(croped_image)
        return tiles



if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('img_out_dir', type=str)

    args = parser.parse_args()

    img_dir = args.img_dir + '/'
    img_out_dir = args.img_out_dir + '/'

    if os.path.exists(img_out_dir):
        shutil.rmtree(img_out_dir)
    os.mkdir(img_out_dir)


    prefix = ''
    img_shape = (2704,3376,3)
    img_tiler = ImageMask2Tiles(img_shape,(450,500))
    for fp in os.listdir(img_dir):
        fn = fp[:-4]
        assert fn+'.jpg' in os.listdir(img_dir)
        img_fp = img_dir+fn+'.jpg'
        print(img_fp)
        img = cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2RGB)
        tiles = img_tiler.processImg(img)
        tiles_keys = tiles.keys()
        for key in tiles_keys:
            fp_write = img_out_dir+fn+'_'+str(key[0])+'_'+str(key[1])+'.jpg'
            print(fp_write)
            cv2.imwrite(fp_write, tiles[key])
        assembled_img = img_tiler.assemblyImg(tiles)
        cv2.imwrite(img_out_dir+fn+'_assembled'+'.jpg', assembled_img)

    
