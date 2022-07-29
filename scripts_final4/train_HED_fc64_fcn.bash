cd ..
CUDA_VISIBLE_DEVICES=4,5 python train.py --exp HED_fc64_fcn --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net HED --backbone 'resnet18' --feature_channel 64 --name_classifier 'FCN'
