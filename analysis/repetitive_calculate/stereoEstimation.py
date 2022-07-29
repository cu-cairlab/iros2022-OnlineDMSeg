import cv2
import numpy as np
from calibration2 import calib
import open3d as o3d
import matplotlib.pyplot as plt

def getDisparity(img0,img1,stereo):
    return stereo.compute(img0,img1)

def loadImg(path0,path1):
    img0 = cv2.imread(path0)

    img1 = cv2.imread(path1)
    #CHECK DATASET
    #img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    #img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE) 

    return img0,img1



def getRepetitiveMask(th_0,bx,Xw,Zw):
    #Zw[Zw>0] = 0.01
    print Xw
    #assert (Zw==100).all()
    print np.min(bx+Xw)
    print "check x"
    print np.max(bx+Xw)
    th = np.arctan(Zw/(bx+Xw))
    th[th<0] = np.pi/2
    print np.max(th)/np.pi*180
    print np.min(th)/np.pi*180
    print "result:"
    print th_0
    print "min:"
    print np.min(th - th_0)
    print "max:"
    print np.max(th - th_0)
    mask = (th - th_0)>0
    return mask

def calculateRepetitive(fp_pair,bx,mask_path,disparity_path=None,rectified0_path=None,rectified1_path=None):
    print "check"

    resizing = 0.5


    path0 = fp_pair[0]
    path1 = fp_pair[1]
    fn0 = path0.split('/')[-1]
    fn1 = path1.split('/')[-1]
    img0,img1 = loadImg(path0,path1)
    img0,img1 = calib.rectifyImage(img0,img1)
    if rectified0_path != None and rectified1_path != None:
        #cv2.imwrite(rectified0_path+fn0[:-4]+'.png',img0,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        #cv2.imwrite(rectified1_path+fn1[:-4]+'.png',img1,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(rectified0_path+fn0[:-4]+'.jpg',img0)
        cv2.imwrite(rectified1_path+fn1[:-4]+'.jpg',img1)
    
    h,w,_= img0.shape
    img0 = cv2.resize(img0,(int(resizing*w),int(resizing*h)), interpolation=cv2.INTER_CUBIC)
    img1 = cv2.resize(img1,(int(resizing*w),int(resizing*h)), interpolation=cv2.INTER_CUBIC)
    #num_disp = max_disp - min_disp # Needs to be divisible by 16#Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= 180, #40
    numDisparities = 16*11, #16*11
    blockSize = 9, #9
    uniquenessRatio = 0,
    speckleWindowSize = 11,
    speckleRange = 100,
    disp12MaxDiff = 100,
    P1 = 500,
    P2 =7000,
    preFilterCap = 5)
    disparity = getDisparity(img1,img0,stereo)
    disparity = disparity.astype(np.float32) / 16.0
    disparity_img = cv2.normalize(disparity, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    #TODO
    if disparity_path != None:
        cv2.imwrite(disparity_path+fn0[:-4]+'.png',disparity_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])

    ptcloud,c,r,depth = calib.project_disp_to_points(disparity,resizing)
    print "*******mean:"
    print np.mean(depth)
    #ptcloud = ptcloud.reshape(-1,3)
    #mask = np.logical_and(ptcloud[:,2]>0.3,ptcloud[:,2]<2.5)
    #ptcloud = ptcloud[mask]



    #depth[depth < 0.7] = 0.7
    #depth[depth > 0.5] = 0.5
    aov = calib.aov_y
    #print aov/np.pi*180.
    th_0 = (np.pi/2)-(aov/2)
    #print th_0
    #bx = 0.33
    Xw = r
    Zw = depth
    mask_repetitive = getRepetitiveMask(th_0,bx,Xw,Zw)
    #print mask_repetitive.shape

    #TODO
    cv2.imwrite(mask_path+fn0[:-4]+'.png',mask_repetitive*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    
    
