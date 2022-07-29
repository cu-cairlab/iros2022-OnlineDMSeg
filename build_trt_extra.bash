echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m1_a1/19022022_190547/best_model_epoch4920.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 1 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m1_a1_nfk.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m1_a1_nfk.trt --fix_kernel_size False

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a1/09022022_190610/best_model_epoch2460.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1_nfk.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a1_nfk.trt --fix_kernel_size False

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m3_a1/19022022_190621/best_model_epoch4620.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 3 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m3_a1_nfk.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m3_a1_nfk.trt --fix_kernel_size False

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m4_a1/19022022_190627/best_model_epoch3090.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 4 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m4_a1_nfk.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m4_a1_nfk.trt --fix_kernel_size False


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/HED_fc1/20022022_212012/best_model_epoch3235.pth' --net HED --backbone 'resnet18' --feature_channel 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc1.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/HED_fc64/20022022_085536/best_model_epoch4845.pth' --net HED --backbone 'resnet18' --feature_channel 64 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 multiLoss<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 multiLoss<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/HED_fc64_multiLoss/20022022_211846/best_model_epoch4520.pth' --net HED --backbone 'resnet18' --feature_channel 64 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64multiLoss.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64multiLoss.trt




echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 fcn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 fcn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#python3 build_trt.py --weight '/media/Data/ignored/HED_fc64_fcn/16022022_051808/best_model_epoch3825.pth' --net HED --backbone 'resnet18' --name_classifier FCN --feature_channel 64 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64fcn.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64fcn.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/Deeplab_rs18_os8/16022022_051800/best_model_epoch2180.pth' --net Deeplabv3 --backbone 'resnet18OS8' --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/deeplabv3rn18os8.trt --trt_int8Path /media/Data/trtModel/int8/deeplabv3rn18os8.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 modified head<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '' --net Deeplabv3 --backbone 'resnet18OS8' --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/deeplabv3rn18os8_modifiedHead.trt --trt_int8Path /media/Data/trtModel/int8/deeplabv3rn18os8_modifiedHead.trt --name_classifier 'deeplab_modified'








