cd ..
CUDA_VISIBLE_DEVICES=2,3 python train.py --exp SFSegNet_resnet --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net SFSegNet --backbone 'resnet'

