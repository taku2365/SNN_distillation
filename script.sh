#!/bin/bash

# Train an ANN for <architecture> on <dataset>
python ann.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-2 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0

# Convert ANN to SNN and perform spike-based backpropagation
python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 20 --leak 1.0 --scaling_factor 0.6 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 50 --devices 2 --default_threshold 1.0 --pretrained_ann './trained_models/ann_vgg16_cifar10.pth'