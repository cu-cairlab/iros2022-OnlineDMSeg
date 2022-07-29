data_path=./20211117/   #PLEASE CHANGE
sudo docker run -it --ipc=host --gpus all -v $PWD:/media -v $data_path:/data dm_project /bin/bash 

