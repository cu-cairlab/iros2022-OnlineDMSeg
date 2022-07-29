#data_path=/media/leo/Data/row013_auto
data_path=/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5/row003_auto
#data_path=/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5//Resnet_Segmentation/data/  #for trainval data
#output_path=${data_path}_out
output_path="/media/leo/dddd21c6-acd6-45cc-a4bf-d796523767ed/row003_auto_out"
weight_path=/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5/assets
#weight_path=/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5/new_weight/newnet_resnet18_64_newHead/06122021_072820/
#weight_path=/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5/HMASSLog_DM
aux_path=/media/leo/5d7aa88a-dacb-4f50-90ec-6af3274316c5/Resnet_Segmentation    #for other networks

sudo docker run --gpus all -it --ipc=host -v $PWD:/media \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $data_path:/data \
    -v $output_path:/output \
    -v $weight_path:/assets \
    -v $aux_path:/media/aux \
    dm_project /bin/bash
