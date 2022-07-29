python draw_masks.py --img 'examples/row01_sec13_right_cam0_frm0296_1_1.jpg' --output ./ignored/visualize --weight './ignored/ignored/newnet_resnet18_64_newHead_m2_a1/04022022_164257/best_model_epoch4470.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1


python draw_masks.py --img 'examples/row01_sec13_right_cam0_frm0296_1_1.jpg' --output ./ignored/visualize --weight './ignored/ignored/HED_fc64_multiLoss/23022022_090420/best_model_epoch3220.pth' --net HED --backbone 'resnet18' --feature_channel 64 --multiLoss True


python draw_masks.py --img 'examples/row01_sec15_right_cam0_frm0350_0_1.jpg' --output ./ignored/visualize --weight './ignored/ignored/HED_fc64_multiLoss/23022022_090420/best_model_epoch3220.pth' --net HED --backbone 'resnet18' --feature_channel 64 --multiLoss True


python draw_masks.py --img 'examples/row01_sec15_left_cam0_frm0488_1_1.jpg' --output ./ignored/visualize_1 --weight './ignored/ignored/newnet_resnet18_64_newHead_m1_a1/05022022_050501/best_model_epoch4695.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 1 --feature_accumumation_start 1

python draw_masks.py --img 'examples/row01_sec15_left_cam0_frm0488_1_1.jpg' --output ./ignored/visualize_2 --weight './ignored/ignored/newnet_resnet18_64_newHead_m2_a1/04022022_164257/best_model_epoch4470.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 1

python draw_masks.py --img 'examples/row01_sec15_left_cam0_frm0488_1_1.jpg' --output ./ignored/visualize_3 --weight './ignored/ignored/newnet_resnet18_64_newHead_m3_a1/05022022_050538/best_model_epoch2045.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 3 --feature_accumumation_start 1

python draw_masks.py --img 'examples/row01_sec15_left_cam0_frm0488_1_1.jpg' --output ./ignored/visualize_4 --weight './ignored/ignored/newnet_resnet18_64_newHead_m4_a1/05022022_050605/best_model_epoch790.pth' --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 4 --feature_accumumation_start 1
