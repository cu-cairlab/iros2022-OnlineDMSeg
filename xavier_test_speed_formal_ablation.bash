echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m1_a1/05022022_050501/best_model_epoch4695.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 1 --feature_accumumation_start 1 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m1_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m1_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a1/04022022_164257/best_model_epoch4470.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m3_a1/05022022_050538/best_model_epoch2045.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 3 --feature_accumumation_start 1 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m3_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m3_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m4_a1/05022022_050605/best_model_epoch790.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 4 --feature_accumumation_start 1 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m4_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m4_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a2m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a2m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a2/04022022_164303/best_model_epoch4960.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 2 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a2.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a2.trt
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a3m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a3m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a3/04022022_164307/best_model_epoch2500.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 3 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a3.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a3.trt
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a4m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a4m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a4/04022022_164312/best_model_epoch3875.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 4 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a4.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a4.trt


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 concat<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 concat <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a1_concat/15022022_012840/best_model_epoch3810.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1 --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1_concat.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a1_concat.trt --use_concat True

