cd ..
CUDA_VISIBLE_DEVICES=6,7 python train.py --exp newnet_resnet18_64_newHead_m2_a4 --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net DeepLabv3_modified_addBN --backbone 'resnet18' --feature_channel 64 --feature_map_size_layer 2 --feature_accumumation_start 4

