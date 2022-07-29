#!/bin/bash
img0_folder=/data/color/cam0_images
img1_folder=/data/color/cam1_images
gps_path=/data/vehicle_pose.txt
row_segmentation=/data/row_segmentation.csv
#row_segmentation=/data/row01.pkl
#weight_dir=/assets/seg_weights/PM_best_checkpoint_ep391.pth
#weight_dir=/media/aux/ignored/dm_new_net_addBN/04112021_053427/best_model_epoch3920.pth
#weight_dir=/assets/best_checkpoint_ep413.pth  #DM
weight_dir=/assets/seg_weights/PM_best_checkpoint_ep312.pth #PM
output_dir=/output

#Repetitive calculate
repetitive_mask_dir=$output_dir/repetitive_mask
disparity=$output_dir/disparity/
rectified0=$output_dir/rectified0/
rectified1=$output_dir/rectified1/
cd repetitive_calculate
source activate py2
python2 main2.py $img0_folder $img1_folder $gps_path 1 ${repetitive_mask_dir}_1/ $disparity $rectified0 $rectified1
python2 main2.py $img0_folder $img1_folder $gps_path 2 ${repetitive_mask_dir}_2/ $disparity $rectified0 $rectified1
python2 main2.py $img0_folder $img1_folder $gps_path 3 ${repetitive_mask_dir}_3/ $disparity $rectified0 $rectified1
python2 main2.py $img0_folder $img1_folder $gps_path 4 ${repetitive_mask_dir}_4/ $disparity $rectified0 $rectified1
cd ..


