import torch
import torch.utils.data as data
import os
import random
import glob
from PIL import Image
from utils import preprocess
import numpy as np
import cv2

class DMData(data.Dataset):
    CLASSES = [
      'BG','DM'
      ]

    def __init__(self, root, train=True, test=False, inference=False,transform=None, target_transform=None, download=False, crop_size=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = 512#crop_size
        self.inference=inference
        ######################################################################
        # Assemble image lists
        ######################################################################
        splits = {'train': 'training',
                  'val': 'validation',
                  'test': 'testing'}

        if not test:
            mode = 'train' if self.train else 'val'
            split_name = splits[mode]
            #split_name=''
            img_ext = 'png'
            mask_ext = 'png'
            img_ext = 'jpg'
            mask_ext = 'png'
            
            img_root = os.path.join(root, split_name, 'Image/')
            mask_root = os.path.join(root, split_name, 'Mask/')
            images=os.listdir(img_root)
            self.images = [img_root+image for image in images]
            #masks=os.listdir(mask_root)
            self.masks = [mask_root+image[:-4]+'.'+mask_ext for image in images]
        if test:
            img_ext = 'png'
            mask_ext = 'png'
            img_ext = 'jpg'
            mask_ext = 'png'
            img_root = os.path.join(root, 'Image/')
            mask_root = os.path.join(root, 'Mask/')
            images=os.listdir(img_root)
            self.images = [img_root+image for image in images]
            self.masks = [mask_root+image[:-4]+'.'+mask_ext for image in images]


    def __getitem__(self, index):
        if not self.inference:
            return self.__getitem_gt__(index)
        else:
            return self.__getitem_no_gt__(index)

    def __getitem_gt__(self, index):
        #print(self.images[index])
        #print(self.masks[index])
        _img = Image.open(self.images[index]).convert('RGB')

        _target = Image.open(self.masks[index])
        _target = _target.crop((0,0,_img.size[0],_img.size[1]))

        #_img, _target = preprocess(_img, _target,
        #                       flip=True if self.train else False,
        #                       scale=(0.5, 2.0) if self.train else None,
        #                       crop=(self.crop_size, self.crop_size))

        _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size) if self.train else (1125, 1352))
                               #crop=(1125,1352) if self.train else (1024,1024))#(1125, 1352))
        #print(self.masks[index])
        #print(_target.size())        
        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)
        #print(_target.size())
        return _img, _target,index


    def __getitem_no_gt__(self, index):
        #print(self.images[index])
        #print(self.masks[index])
        _img = Image.open(self.images[index]).convert('RGB')



        #_img, _target = preprocess(_img, _target,
        #                       flip=True if self.train else False,
        #                       scale=(0.5, 2.0) if self.train else None,
        #                       crop=(self.crop_size, self.crop_size))

        _img = preprocess_image(_img,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size) if self.train else (1125, 1352))
        #print(self.masks[index])
        #print(_target.size())        
        if self.transform is not None:
            _img = self.transform(_img)


        #print(_target.size())
        return _img,index



    def __len__(self):
        return len(self.images)

    def download(self):
        raise NotImplementedError('Automatic download not yet implemented.')


if __name__ == "__main__":
    root = '/media/data/DM4/'
    dataset = DMData(root)
    idx = 0
    for img,mask in dataset:
        idx= idx + 1
        img = (img.permute(1, 2, 0).data.cpu().numpy()*255).squeeze().astype(np.int32)
        
        mask = mask.data.cpu().numpy().squeeze().astype(np.uint8)
        #print(img.shape)
        #print(mask.shape)
        redImg = np.zeros(img.shape, img.dtype)
        redImg[:,:] = (0, 0, 255)
        redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
        #print(img.shape)
        #print(redMask.shape)
        img = img.copy().astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB ).astype(np.int32)
        cv2.imwrite('/media/output/'+str(idx)+'_img.jpg',img)
        cv2.addWeighted(redMask, 0.5, img, 1, 0, img)
        cv2.imwrite('/media/output/'+str(idx)+'_mask.jpg',img)

        #img.save('/media/output/'+str(idx)+'_img.jpg')
        #mask.save('/media/output/'+str(idx)+'_mask.png')
