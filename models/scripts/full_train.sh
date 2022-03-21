#!/bin/sh
gpu=( $@ )
gpu_num=$#
echo $gpu_num

data=coco.yaml
ID=20210619-203749
yamlID=49
project=runs/train-$ID
cfg=$project/exp/genotypes/$yamlID.yaml
LOG_DIR=test_logs/yolov5x
CONFIG=""
#CONFIG="--multi-scale"
train_type=from_scratch  # load_from_search / from_scratch

BATCHSIZE=64

gpustr=${gpu[0]}
for i in ${gpu[@]:1:$gpu_num-1};do
gpustr=$gpustr','$i;
done
echo $gpustr

CUDA_VISIBLE_DEVICES=$gpustr python train.py $CONFIG --data $data --cfg $cfg --project $project$CONFIG --name full_train --weights '' --train_type $train_type --batch-size $BATCHSIZE > $LOG_DIR/$ID-yaml$yamlID-$train_type$CONFIG-half.log 2>&1 &

#python train.py --data coco.yaml --cfg runs/train-20210408-105854/exp/genotypes/96.yaml --weights '' --project runs/train-20210408-105854 --name debug

