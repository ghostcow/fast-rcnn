#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/imagenet_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 0 \
	--solver models/VGG16/solver.prototxt \
	--weights output/default/ILSVRC2014_train/vgg16_fast_rcnn_iter_120000.caffemodel \
	--iters 60000
#	--weights data/imagenet_models/VGG16.v2.caffemodel
#	--iters 200000

time ./tools/test_net.py --def models/VGG16/test.prototxt \
	--imdb ILSVRC2014_val  \
	--net output/default/ILSVRC2014_train/vgg16_fast_rcnn_iter_60000.caffemodel
