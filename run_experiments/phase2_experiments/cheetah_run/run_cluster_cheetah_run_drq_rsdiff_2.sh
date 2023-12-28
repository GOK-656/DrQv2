#!/bin/bash

cd /bigdata/users/ve490-fall23/lyf/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=cheetah_run
seed=2

echo "start running $tag with seed $seed"
python train.py aug_type=19 task=cheetah_run experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
