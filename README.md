# EAutoDet
Implementation of [EAutoDet](), an efficient NAS method for Object Detection. Code is developed based on [YOLOV5](https://github.com/ultralytics/yolov5)

# Environment

The environment of this project is the same as [YOLOv5](https://github.com/ultralytics/yolov5). Also, you can build a singularity image by:

`singularity build <name/of/image>.sig envs/singularity.def`

# Performance

We define four search spaces, denoted as EAutoDet-s (small), EAutoDet-m (mediun), EAutoDet-l (large), and EAutoDet-x (extra-large). We also train YOLOv5 from scratch and compare with our discovered models. FPS are evaluated on the [Darknet platform](https://github.com/pjreddie/darknet), which is written in C and CUDA.

| Model         | mAP(0.5:0.95) | Params (M) | FPS |
| ------------------ |---------- |----------- |----|
| YOLOv5-s   |   36.9     | 7.3      |113|
| YOLOv5-m   |   43.9     | 21.4     |88|
| YOLOv5-l   |   46.8     | 47.1     |59|
| YOLOv5-x   |   49.1     | 87.8     |43|
| EAutoDet-s   |   40.1   | 9.1      |120|
| EAutoDet-m   |   45.2   | 28.1     |70|
| EAutoDet-l   |   47.9   | 34.4     |59|
| EAutoDet-x   |   49.2   | 86.0     |41

# Quick Start
## How to search an architecture
You can search an EAutoDet-s on the 0-th GPU for 50 epochs by running the following codes:

`bash scripts/search.sh 0`

If you want to search on other spaces, you can change the value of `cfg_file` in `scripts/bash.sh` an run the above codes.

If you want to search with a large batch size on multiple GPUs, ruu the following codes:

`bash scripts/search.sh 0 1 2 3`

## How to train the discovered architecture
After searching, the code will save the genotype of discovered architectures in the directory `runs/train-$ID/exp/genotypes/`, where $ID is the timestamp of your search process. Then you can evluate the discovered architecture by training the discovered architecture for 300 epochs from scratch. Run the following codes to train on a single GPU.

`bash scritps/full_train.sh 0`

Notice that before running the above codes, you should change `ID` as the timestamp of your search process.

If you want to train with a large batch sizei on multiple GPUs, run the following codes:

`bash scripts/full_train.sh 0 1 2 3`

## How to evaluate the trained model
You can the following codes to test on the test set of COCO:

`bash scripts/eval.sh 0`

Then you can submit the results (json file) to [COCO evaluation website](https://competitions.codalab.org/competitions/20794#learn_the_details)

## How to evaluate FPS
You can change the discovered architecture configuration (yaml file) to Darknet configuration and then test on Darknet platform, the codes are coming soon.


# Citations

