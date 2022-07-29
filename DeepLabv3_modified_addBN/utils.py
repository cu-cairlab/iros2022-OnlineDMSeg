import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time



def randomAffine(imgs):
  rotation = random.uniform(-15., 15.)
  shear = random.uniform(-15., 15.)
  scale = 1
  x = random.uniform(-50., 50.)
  y = random.uniform(-50, 50.)
  translate = [x,y]
  returnImgs = []
  for img in imgs:
    returnImgs.append(transforms.functional.affine(img, angle=rotation, translate=translate, scale=scale, shear=shear, fill=255))
  return returnImgs

class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)


def preprocess(image, mask, flip=False, scale=None, crop=None):

  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

  if flip:
    image,mask = randomAffine([image,mask]) 


  if scale:
    w, h = image.size
    rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    image = image.resize(new_size, Image.ANTIALIAS)
    mask = mask.resize(new_size, Image.NEAREST) 

  #image.save('img.png')
  #mask.save('mask.png')
  #exit(0)


  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  #print(np.unique(np.array(image)))
  image = data_transforms(image)
  #print(torch.unique(image))
  mask = torch.LongTensor(np.array(mask).astype(np.int64))


  if crop:
    h, w = image.shape[1], image.shape[2]
    pad_tb = max(0, crop[0] - h)
    pad_lr = max(0, crop[1] - w)
    image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
    mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

    h, w = image.shape[1], image.shape[2]
    i = random.randint(0, h - crop[0])
    j = random.randint(0, w - crop[1])
    #i = 0
    #j = 0
    image = image[:, i:i + crop[0], j:j + crop[1]]
    mask = mask[i:i + crop[0], j:j + crop[1]]

  return image, mask

def preprocess_image(image, flip=False, scale=None, crop=None):

  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)

  if flip:
    image = randomAffine([image]) 


  if scale:
    w, h = image.size
    rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    image = image.resize(new_size, Image.ANTIALIAS)


  #image.save('img.png')
  #mask.save('mask.png')
  #exit(0)


  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  #print(np.unique(np.array(image)))
  image = data_transforms(image)
  #print(torch.unique(image))



  if crop:
    h, w = image.shape[1], image.shape[2]
    pad_tb = max(0, crop[0] - h)
    pad_lr = max(0, crop[1] - w)
    image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)


    h, w = image.shape[1], image.shape[2]
    i = random.randint(0, h - crop[0])
    j = random.randint(0, w - crop[1])
    if not flip:
        i = 0
        j = 0
    image = image[:, i:i + crop[0], j:j + crop[1]]


  return image

