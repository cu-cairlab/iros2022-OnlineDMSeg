import stereoEstimation
import readGPS
import os
import shutil
import argparse



if __name__=="__main__":
    print "check"


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('img0_folder', type=str, default="/home/leo/Nvidia_SS/semantic-segmentation2/logs/eval_DM/dexterous-bee_2021.04.21_13.32/best_images/")
    parser.add_argument('img1_folder', type=str, default="/home/leo/Nvidia_SS/semantic-segmentation2/logs/eval_DM/dexterous-bee_2021.04.21_13.32/best_images/")
    parser.add_argument('gps_path', type=str, default='../Data/infection_mask/')
    parser.add_argument('downsample', type=int, default=1)
    parser.add_argument('output', type=str, default='../Data/infection_mask/')
    parser.add_argument('disparity', type=str, default='../Data/infection_mask/')
    parser.add_argument('rectified0', type=str, default='../Data/infection_mask/')
    parser.add_argument('rectified1', type=str, default='../Data/infection_mask/')
    args = parser.parse_args()

    '''
    dataFolder = "/data/"
    gps_fp = dataFolder + "/vehicle_pose.txt"
    imgFolder = dataFolder + "/color/"

    mask_path="/data/mask/"
    disparity_path="/data/disp/"
    rectified_path="/data/rect/"
    '''
    #dataFolder = "/data/"
    downsample = args.downsample
    gps_fp = args.gps_path
    img0_folder = args.img0_folder
    img1_folder = args.img1_folder

    mask_path=args.output
    disparity_path=args.disparity
    rectified0_path=args.rectified0
    rectified1_path=args.rectified1

    
    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)
    if os.path.exists(disparity_path):
        shutil.rmtree(disparity_path)
    if os.path.exists(rectified0_path):
        shutil.rmtree(rectified0_path)
    if os.path.exists(rectified1_path):
        shutil.rmtree(rectified1_path)

    os.mkdir(mask_path)
    os.mkdir(disparity_path)
    os.mkdir(rectified0_path)
    os.mkdir(rectified1_path)



    rowBxs = readGPS.getBx(gps_fp,downsample)
    print len(rowBxs)
    imgs_cam0 = os.listdir(img0_folder)
    imgs_cam1 = os.listdir(img1_folder)
    imgs_cam0.sort()
    imgs_cam1.sort()
    imgs_cam0 = imgs_cam0[::downsample]
    imgs_cam1 = imgs_cam1[::downsample]    


    frameIDs = []
    imgBxs = []
    assert rowBxs[0] == 0
    for fn in imgs_cam0:
        #frameID = int(fn.split('_')[-1][3:-4])
        frameID = int(fn[5:-4])
        frameIDs.append(frameID)
        #print frameID+1
        if frameID+1 < len(rowBxs):
            imgBxs.append(rowBxs[frameID+1])
    assert len(frameIDs) == len(set(frameIDs)) #check repetitive
    assert(len(imgs_cam0) == len(imgs_cam1))
    #assert(len(imgs_cam0) == len(imgBxs))
    print len(imgs_cam0)
    row = [imgs_cam0,imgs_cam1,rowBxs]
    #print imgs_cam0

    for imgID in range(len(row[0])-1):    #skip last frame. Will be replaced by 0
        imgs_cam0,imgs_cam1,imgBxs = row
        print (imgs_cam0[imgID],imgs_cam1[imgID],imgBxs[imgID])
        stereoEstimation.calculateRepetitive((img0_folder+'/'+imgs_cam0[imgID],img1_folder+'/'+imgs_cam1[imgID]),imgBxs[imgID],mask_path,disparity_path,rectified0_path,rectified1_path)

    '''
    assert(len(imgs_cam0) == len(imgs_cam1))
    for i in range(len(imgs_cam0)):
        img_cam0 = imgs_cam0[i]
        img_cam1 = imgs_cam1[i]
        assert(img_cam0.split('_')[0] == img_cam1.split('_')[0])
        assert(img_cam0.split('_')[1] == img_cam1.split('_')[1])
        assert(img_cam0.split('_')[2] == img_cam1.split('_')[2])
        assert(img_cam0.split('_')[4] == img_cam1.split('_')[4])
    '''
        
    







    
    
