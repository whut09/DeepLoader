#!/bin/bash

lfw_pair=pairs.txt
lfw_dir=lfw

# caffe
#python -u eval/run_verify.py --type=caffe --test_set=lfw --data=$lfw_pair --prefix=lfw_dir --model_name=centerface
# tensorflow
#python -u eval/run_verify.py --type=tensorflow --test_set=lfw --data=$lfw_pair --prefix=lfw_dir --model_name=centre --model_path=path/to/model.ckpt
# mxnet
python -u eval/run_verify.py --type=mxnet --test_set=lfw --data=$lfw_pair --prefix=lfw_dir --model_name=centre --model_path=path/to/model --do_norm=False --embed_name=fc1_output --image_size=112,112
