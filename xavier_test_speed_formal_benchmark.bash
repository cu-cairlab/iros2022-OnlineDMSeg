echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attanet resnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attanet resnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/AttanetRN18/08022022_212634/best_model_epoch2865.pth' --net Attanet --trt_fp16Path /media/Data/trtModel/fp16/AttanetRN18.trt --trt_int8Path /media/Data/trtModel/int8/AttanetRN18.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SFSegnet resnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SFSegnet resnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight './media/Data/ignored/SFSegNet_resnet/24022022_224332/best_model_epoch4930.pth' --trt 0 --net SFSegNet --backbone 'resnet'

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SFSegnet dfnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SFSegnet dfnet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight './ignored/ignored/SFSegNet_df2/24022022_224454/best_model_epoch1700.pth' --trt 0 --net SFSegNet --backbone 'dfnet'


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/HED_fc1/20022022_212012/best_model_epoch3235.pth' --net HED --backbone 'resnet18' --feature_channel 1 --trt_fp16Path /media/Data/trtModel/fp16/HEDfc1.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc1.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/HED_fc64/23022022_090420/best_model_epoch3220.pth' --net HED --backbone 'resnet18' --feature_channel 64 --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64.trt



echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 multiLoss<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 multiLoss<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/HED_fc64_multiLoss/23022022_090746/best_model_epoch4690.pth' --net HED --backbone 'resnet18' --feature_channel 64 --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64multiLoss.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64multiLoss.trt




echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 fcn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HED fc64 fcn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/HED_fc64_fcn/24022022_005825/best_model_epoch4730.pth' --net HED --backbone 'resnet18' --name_classifier FCN --feature_channel 64 --trt_fp16Path /media/Data/trtModel/fp16/HEDfc64fcn.trt --trt_int8Path /media/Data/trtModel/int8/HEDfc64fcn.trt

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 original<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 original <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/Deeplab_rs18_os8/16022022_051800/best_model_epoch2180.pth' --net Deeplabv3 --backbone 'resnet18OS8' --trt_fp16Path /media/Data/trtModel/fp16/deeplabv3rn18os8.trt --trt_int8Path /media/Data/trtModel/int8/deeplabv3rn18os8.trt --name_classifier 'deeplab'

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 modified<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#echo -e ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deeplab rn18 os8 modified <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
python3 Xavier_SpeedTest.py --weight '/media/Data/ignored/Deeplab_rs18_os8_mh/21022022_151250/best_model_epoch3525.pth' --net Deeplabv3 --backbone 'resnet18OS8' --trt_fp16Path /media/Data/trtModel/fp16/deeplabv3rn18os8_mh.trt --trt_int8Path /media/Data/trtModel/int8/deeplabv3rn18os8_mh.trt --name_classifier 'deeplab_modified'




