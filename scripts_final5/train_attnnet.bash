cd ..
CUDA_VISIBLE_DEVICES=4,5 python train.py --exp AttanetRN18 --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net Attanet

