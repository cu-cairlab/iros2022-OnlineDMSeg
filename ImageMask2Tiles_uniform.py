import PIL.Image
import numpy as np
import cv2
import json
import os
import argparse

class IMAGE:
    pass

class TILE:
    pass

class BBOX:
    pass



def assemblyImg(result_tiles):
    max_x = 1
    max_y = 2
    #originalShape = (3376, 2704, 3)
    originalShape = (3376, 2704)
    assembledImg = np.zeros(originalShape)
    x_pointer = 0
    y_pointer = 0
    for x in range(max_x+1):
        y_pointer = 0
        for y in range(max_y+1):
            #tile = cv2.imread(imgDir+imgFileDict[(x,y)])
            tile = result_tiles[(x,y)]
            #print np.unique(tile)
            h_t,w_t = tile.shape
            assembledImg[y_pointer:y_pointer+h_t,x_pointer:x_pointer+w_t] = tile
            #print (x_pointer,y_pointer)
            y_pointer = y_pointer+h_t
        x_pointer = x_pointer+w_t

    assert (x_pointer - originalShape[1]) <=1
    assert (y_pointer - originalShape[0]) <=1

    return assembledImg



            
    

def getTiles(im,M,N):
    #print("getting tiles")
    h,w,c = im.shape
    tiles = []
    last_x = 0
    last_y = 0
    '''
    print(w)
    print(M)
    print(M/2)
    print(w-M+M/2)
    '''
    M = int(M)
    N = int(N)
    for x in range(0,int(w-M+M/2),int(M)):
        last_x = x
        for y in range(0,int(h-N+N/2),int(N)):
            tile = TILE()
            #tile.image = im[x-inflation_x:x+M+inflation_x,y-inflation_y:y+N+inflation_y]
            tile.image = im[y:y+N,x:x+M]
            #print((x,y))
            #tile.to2io = (pad_left+inflation_x-x,pad_top+inflation_y-y)
            tiles.append(tile)
            last_y = y


    '''
        tile = TILE()
        tile.image = im[last_y+N:,x:x+M]
        tiles.append(tile)
        print (x,last_y+N)

    for y in range(0,h-N-1,N):
        tile = TILE()
            #tile.image = im[x-inflation_x:x+M+inflation_x,y-inflation_y:y+N+inflation_y]
        tile.image = im[y:y+N,last_x+M:]
            #tile.to2io = (pad_left+inflation_x-x,pad_top+inflation_y-y)
        tiles.append(tile)
        last_y = y
        print (last_x+M,y)
    tile = TILE()
    tile.image = im[last_y+N:,last_x+M:]
    tiles.append(tile)
    '''
    #print((last_x+M,last_y+N))
    return tiles




     
   
def processImg(image):
    #M = 563+64+50 #x
    #N = 451+64+50 #y
    #M = M * 2
    #N = N * 2
    #inflation_x = 27+32 + 32
    #inflation_y = 11+64 + 32
    inflation_x = 0
    inflation_y = 0

    #image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
    #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
    #cv2.imwrite("testout2/"+fn+'.png',image,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    h,w,c = image.shape
    #print(image.shape)
    M = w/2
    N = h/3
    #M = w/4
    #N = h/6
    tiles = getTiles(image,M,N)
    return tiles



if __name__=="__main__":

    img_dir = '/home/leo/Downloads/2019_dataset/images/'
    img_out_dir = 'Image_Out3/'

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('img_out_dir', type=str)

    args = parser.parse_args()

    img_dir = args.img_dir
    img_out_dir = args.img_out_dir

    if os.path.exists(img_out_dir):
        shutil.rmtree(img_out_dir)
    os.mkdir(img_out_dir)


    prefix = ''
    for fp in os.listdir(img_dir):
        fn = fp[:-4]
        assert fn+'.jpg' in os.listdir(img_dir)
        img_fp = img_dir+fn+'.jpg'
        processImg(img_fp,fn,prefix,img_out_dir)

    
