#!/bin/sh
gpu=( $@ )
gpu_num=$#
echo $gpu_num

data=coco.yaml
cfg_file=EAutoDet-s
cfg=models/search_cfg/$cfg_file.yaml
LOG_DIR=logs

BATCHSIZE=80

gpustr=${gpu[0]}
for i in ${gpu[@]:1:${gpu_num}-1};do
gpustr=$gpustr','$i;
done
echo $gpustr

CUDA_VISIBLE_DEVICES=$gpustr python -u train_search.py --data $data --cfg $cfg --weights '' --epochs 50 --batch-size $BATCHSIZE > $LOG_DIR/${cfg_file}_test1.log 2>&1 &

