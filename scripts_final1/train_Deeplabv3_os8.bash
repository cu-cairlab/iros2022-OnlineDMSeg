cd ..
CUDA_VISIBLE_DEVICES=2,3 python train.py --exp Deeplab_rs18_os8 --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net Deeplabv3 --backbone 'resnet18OS8' --name_classifier 'deeplab'

