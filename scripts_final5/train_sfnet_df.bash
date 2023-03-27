cd ..
CUDA_VISIBLE_DEVICES=0,1 python train.py --exp SFSegNet_df2 --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net SFSegNet --backbone 'df2'

