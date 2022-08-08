# Near-real time Grape Downy Mildew Segmentation


Repository contains codes and datasets to reproduce results in the IROS 2022 submission on "Near Real-Time Vineyard Downy Mildew Detection and Severity Estimation"

This codebase is an undergoing project. Improvements will be continuesly made.\
Todo: \
-Training instruction \
-Testing instruction \
-Deployment instuction \
-Test on different platforms 

Training
1. Download the data from the link: https://cornell.box.com/s/3i2rod89mwb17apjeo103v39n33aaaq7

2. Build the dockerfile in this repository by running:
"sudo docker build -t dm_project ./dockerfile" \

3. Change data_path to path of training dataset. Activate docker image by running:
bash docker.bash

4. Run one of the scripts under scripts_final* directory. Directory numbers are associated with table numbers in the paper.
\
Testing On Xavier:
1. Install JetPack 4.6 rev.3

2. Install Pytorch, torchvision

3. Install torch2trt

4. Mount external ssd with path /media/Data

5. Download the testing data from the link: https://cornell.box.com/s/4dsgv2xr9nubjxwqfs38pdf1at6e0nvu to /media/Data

6. Download the testing weight from the link: https://cornell.box.com/s/l691wfgaz81i5qlhicnz2g8g7j9na1o5 to /media/Data

7. Run formal_tests_ablation.bash or formal_tests_benchmark.bash
