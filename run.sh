#!/bin/bash
if [ "$1" = "retrain_distillation" ]
then 
    echo "Retrain !!"
    python snn_skd.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation False  --dataset CIFAR100 --timesteps 5 --input_compress_rate 0.45 --rank_reduce True \
    --vgg_after_distillation=./experiments/sskd_student_VGG16_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_wrn_40_2_seed0_retrain/ckpt/student_best.pth\
    --vgg_stdb_after_distillation=./experiments/sskd_student_VGG_SNN_STDB_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_ResNet50_seed0_CIFAR100_timesteps5/ckpt/student_best.pth
elif [ "$1" = "test" ]
then
    echo "test !!"

elif [ "$1" = "float_calculation" ]
then
    echo "Float Cal!!"
    python3 spike_rate_visualize.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.00005 --gpu-id "0" --milestones 5 10 50 --epoch 15  --after_distillation Ture --dataset CIFAR100 --timesteps 5  \
    --vgg_after_distillation=./experiments/sskd_student_VGG16_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_wrn_40_2_seed0_retrain/ckpt/student_best.pth \
    --vgg_stdb_after_distillation=./experiments/sskd_student_VGG_SNN_STDB_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_ResNet50_seed0_CIFAR100_timesteps5/ckpt/student_best.pth 

else
    echo "index not found"
fi


