echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m1_a1/05022022_050501/best_model_epoch4695.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 1 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m1_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m1_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a1/04022022_164257/best_model_epoch4470.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m3_a1/05022022_050538/best_model_epoch2045.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 3 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m3_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m3_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m4_a1/05022022_050605/best_model_epoch790.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 4 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m4_a1.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m4_a1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a2m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a2m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a2/04022022_164303/best_model_epoch4960.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 2 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a2.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a2.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a3m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a3m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a3/04022022_164307/best_model_epoch2500.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 3 --calib_dataPath ~/20211117/DM4/testing --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a3.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a3.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a4m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a4m2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a4/04022022_164312/best_model_epoch3875.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 4 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a4.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a4.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 concat<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab Modified a1m2 concat <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

#python3 build_trt.py --weight '/media/Data/ignored/newnet_resnet18_64_newHead_m2_a1_concat/15022022_012840/best_model_epoch3810.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1_concat.trt --trt_int8Path /media/Data/trtModel/int8/newnet_resnet18_64_newHead_m2_a1_concat.trt --use_concat True



echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attanet resnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attanet resnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/AttanetRN18/08022022_212634/best_model_epoch2865.pth' --net Attanet --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/AttanetRN18.trt --trt_int8Path /media/Data/trtModel/int8/AttanetRN18.trt


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#python3 build_trt.py --weight '/media/Data/ignored/HED_fc1/15022022_082625/best_model_epoch2995.pth' --net HED --backbone 'resnet18' --feature_channel 1 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc1.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/HED_fc64/23022022_090420/best_model_epoch3220.pth' --net HED --backbone 'resnet18' --feature_channel 64 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 multiLoss<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 multiLoss<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/HED_fc64_multiLoss/23022022_090746/best_model_epoch4690.pth' --net HED --backbone 'resnet18' --feature_channel 64 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64multiLoss.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64multiLoss.trt




echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 fcn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 fcn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/HED_fc64_fcn/24022022_005825/best_model_epoch4730.pth' --net HED --backbone 'resnet18' --name_classifier FCN --feature_channel 64 --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64fcn.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64fcn.trt


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 original<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 original<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/Deeplab_rs18_os8/16022022_051800/best_model_epoch2180.pth' --net Deeplabv3 --backbone 'resnet18OS8' --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/deeplabv3rn18os8.trt --trt_int8Path /media/Data/trtModel/int8/deeplabv3rn18os8.trt --name_classifier 'deeplab'


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 modified<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 modified<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 build_trt.py --weight '/media/Data/ignored/Deeplab_rs18_os8_mh/21022022_151250/best_model_epoch3525.pth' --net Deeplabv3 --backbone 'resnet18OS8' --calib_dataPath ~/20211117/DM4/testing --trt_fp16Path /media/Data/trtModel/fp16/deeplabv3rn18os8_mh.trt --trt_int8Path /media/Data/trtModel/int8/deeplabv3rn18os8_mh.trt --name_classifier 'deeplab_modified'






