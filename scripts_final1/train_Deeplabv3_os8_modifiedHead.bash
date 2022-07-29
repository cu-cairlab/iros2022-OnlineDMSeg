cd ..
CUDA_VISIBLE_DEVICES=0,1 python train.py --exp Deeplab_rs18_os8_mh --epochs 5000 --batch_size 10 --base_lr 0.0001 --workers 8 --net Deeplabv3 --backbone 'resnet18OS8' --name_classifier 'deeplab_modified'

