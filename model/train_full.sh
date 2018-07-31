#!/usr/bin/env sh

CAFFE_ROOT=../../../
cd $CAFFE_ROOT
build/tools/compute_image_mean.bin -backend=leveldb ./examples/cnn_particlefilter/train_leveldb ./examples/cnn_particlefilter/model/mean.binaryproto
build/tools/caffe train --solver examples/cnn_particlefilter/model/solver.prototxt
