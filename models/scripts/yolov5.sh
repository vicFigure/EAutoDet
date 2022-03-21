CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64 </dev/null> logs/yolov5.log 2>&1 &

