cd ..
CUDA_VISIBLE_DEVICES=0,1 python train.py --exp HED_fc64_multiLoss --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net HED --backbone 'resnet18' --feature_channel 64 --multiLoss True
