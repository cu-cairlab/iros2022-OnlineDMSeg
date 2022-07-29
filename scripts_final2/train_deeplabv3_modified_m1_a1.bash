cd ..
CUDA_VISIBLE_DEVICES=0,1 python train.py --exp newnet_resnet18_64_newHead_m1_a1 --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 1 --feature_accumumation_start 1

