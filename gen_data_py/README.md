# Python codes for simulating single-photon lidar data

In the terminal, run

`bash download_data.sh 0`
`bash download_data.sh 1`

Then, it will download middlebury dataset and MPI dataset.

Next, run

`python create_data_middlebury.py 32 64`

, which will create dataset with `ppp=32` and `sbr=64`. Please check the inside of the python file. For example, you can change btrain=0 or btain=1 for generating training dataset or test dataset.
