#python snn_sskd.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch wrn_16_2 --lr 0.001 --gpu-id 0
# python snn_visualize.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch VGG16 --lr 0.0001 --gpu-id "0,1" --epoch 500 --milestones [250,350,400]
# python snn_visualize.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation Ture --dataset CIFAR100 --timesteps 5 
# python snn_visualize.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation Ture --dataset CIFAR100 --timesteps 5 --input_compress_num 1 --input_compress_rate 0.2 --rank_reduce True
# python snn_visualize.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.00005 --gpu-id "0,1" --milestones 5 10 50 --epoch 15  --after_distillation Ture --dataset CIFAR100 --timesteps 5 --input_compress_num 1 --input_compress_rate 0.2 --rank_reduce True  
# python snn_visualize.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30  --epoch 100  --after_distillation Ture --dataset CIFAR100 
# python snn_visualize.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR10/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 250 350 400 --epoch 500   --dataset CIFAR10 
# sskd_student_VGG_SNN_STDB_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_wrn_40_2_seed0 74.63%  t wideresnet40_2   VGG after_distillation 75~

import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
# from wrapper import wrapper
from cifar import CIFAR100_aug,CIFAR10_aug
from torchvision.datasets import CIFAR100,CIFAR10
from self_models import *
from self_models import model_dict



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[5,10,15])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--ce-weight', type=float, default=0.1) # cross-entropy
parser.add_argument('--kd-weight', type=float, default=0.9) # knowledge distillation
parser.add_argument('--tf-weight', type=float, default=2.7) # transformation
parser.add_argument('--ss-weight', type=float, default=10.0) # self-supervision

parser.add_argument('--kd-T', type=float, default=4.0) # temperature in KD
parser.add_argument('--tf-T', type=float, default=4.0) # temperature in LT
parser.add_argument('--ss-T', type=float, default=0.5) # temperature in SS

parser.add_argument('--ratio-tf', type=float, default=1.0) # keep how many wrong predictions of LT
parser.add_argument('--ratio-ss', type=float, default=0.75) # keep how many wrong predictions of SS
parser.add_argument('--s-arch', type=str) # student architecture
parser.add_argument('--t-path', type=str) # teacher checkpoint path
parser.add_argument('--s-path', type=str) # teacher checkpoint path
parser.add_argument('--retrain', type=bool,default=False) # teacher checkpoint path
parser.add_argument('--after_distillation', type=bool,default=False) # teacher checkpoint path
parser.add_argument('--self_distillation', type=bool,default=False) # teacher checkpoint path
parser.add_argument('--dataset',  default='CIFAR10',type=str, help='dataset name', choices=['CIFAR10','CIFAR100'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=str, default=0)
parser.add_argument('--timesteps', default=5, type=int, help='simulation timesteps')
parser.add_argument('--input_compress_num', default=0,   type=int )
parser.add_argument('--input_compress_rate', default=0.45,  type=float)
parser.add_argument('--rank_reduce', default=False,   type=bool )
parser.add_argument('--retrain_iter', default=15,   type=int)


args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


t_name = osp.abspath(args.t_path).split('/')[-1]
t_arch = '_'.join(t_name.split('_')[1:-2])
exp_name = 'sskd_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}+{}_seed{}_{}_timesteps{}'.format(\
            args.s_arch, \
            args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight, \
            args.kd_T, args.tf_T, args.ss_T, \
            args.ratio_tf, args.ratio_ss, \
            args.seed, t_name,args.timesteps)
exp_path_t = './experiments/{}'.format(exp_name)
print(exp_path_t)
os.makedirs(exp_path_t, exist_ok=True)

vgg_after_distillation_CIFAR100= "./experiments/sskd_student_VGG16_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_wrn_40_2_seed0_retrain/ckpt/student_best.pth"
vgg_stdb_after_distillation_CIFAR100= "./experiments/sskd_student_VGG_SNN_STDB_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_ResNet50_seed0_CIFAR100_timesteps5/ckpt/student_best.pth"

if args.dataset == 'CIFAR100':
    normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    num_classes      = 100 
elif args.dataset == 'CIFAR10':
    normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    num_classes      = 10


transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
   normalize,
])

num = 0
feature_result = torch.tensor(0.)
total = torch.tensor(0.)



if args.dataset == "CIFAR100":
    trainset = CIFAR100('./data', train=True, transform=transform_train,download=True)
    valset = CIFAR100('./data', train=False, transform=transform_test,download=True)
    trainset = CIFAR100_aug('./data', train=True, transform=transform_train)
    valset = CIFAR100_aug('./data', train=False, transform=transform_test)
else:
    trainset = CIFAR10('./data', train=True, transform=transform_train,download=True)
    valset = CIFAR10('./data', train=False, transform=transform_test,download=True)
    trainset = CIFAR10_aug('./data', train=True, transform=transform_train)
    valset = CIFAR10_aug('./data', train=False, transform=transform_test)


train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

ckpt_path = osp.join(args.t_path,'ckpt/best.pth') 

t_model = model_dict[t_arch](num_classes=num_classes).cuda() 
state_dict = torch.load(ckpt_path)['state_dict'] 
# state_dict = torch.load("/home/takuya-s/SNN_test/diet_snn/trained_models/ann_wideresnet40-2_cifar100_best_model.pth")['state_dict'] 
t_model.load_state_dict(state_dict) 

# t_optimizer = optim.SGD([{'params':t_model.backbone.parameters(), 'lr':0.0},
#                         {'params':t_model.proj_head.parameters(), 'lr':args.t_lr}],
#                         momentum=args.momentum, weight_decay=args.weight_decay)]

t_optimizer = optim.Adam(t_model.parameters(),
                         lr=args.lr,
                         amsgrad=True, weight_decay=0, betas=(0.9,0.999))
t_model.eval()
t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)

logger = SummaryWriter(osp.join(exp_path_t, 'events'))

acc_record = AverageMeter()
loss_record = AverageMeter()
start = time.time()

acc_random = []
acc_proposion = []
x_channel_ax    = [] 





# net_name = "RESNET20_BATCH_NORM",labels=100, timesteps=5,dropout=0.3, dataset="CIFAR100",t_divede=5)
if args.s_arch == "VGG_SNN_STDB":
    # if args.retrain:
    #     state = torch.load("{}/ckpt/student_best.pth".format(exp_path_t), map_location='cpu') 
    s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset)
    s_model = nn.DataParallel(s_model)
 

    if args.after_distillation:
        if args.dataset == "CIFAR100":
            state = torch.load(vgg_after_distillation_CIFAR100,map_location='cpu')
        elif args.dataset == "CIFAR10":
            state = torch.load(vgg_after_distillation_CIFAR100,map_location='cpu')
    
    else:
        if args.dataset == "CIFAR100":
            state = torch.load("trained_models/ann_vgg16_cifar100_best_model.pth", map_location='cpu') 
        elif args.dataset == "CIFAR10":
            state = torch.load("trained_models/ann_vgg16_cifar10_best_model1.pth", map_location='cpu') 

    thresholds = state['thresholds']
    print(thresholds)

    missing_keys, unexpected_keys = s_model.load_state_dict(state['state_dict'],strict=False)
    print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))

    s_model.cuda()

    s_model.eval()

    
    
    

    if args.rank_reduce:
        # s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset)
        state = torch.load(vgg_stdb_after_distillation_CIFAR100, map_location='cpu') 
        missing_keys, unexpected_keys = s_model.load_state_dict(state['state_dict'],strict=False)
        s_model.module.rank_reduce = False
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))

        s_model.cuda()
        s_model.eval()
        # optimizer =  optim.Adam(s_model.parameters(),
        #                 lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))        

        acc_record = AverageMeter()
        loss_record = AverageMeter()




        s_model_dict = s_model.state_dict()
    

        input_compress_tmp = 0


        for _ in range(s_model.module.features[0].weight.size(0)):
            input_compress_tmp += 2
            x_channel_ax.append(input_compress_tmp)
            s_model_rank = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=input_compress_tmp)
            s_model_rank = nn.DataParallel(s_model_rank)  
            s_model_rank.cuda()
            s_model_rank1 = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=input_compress_tmp)
            s_model_rank1 = nn.DataParallel(s_model_rank1)  
            name = osp.join(exp_path_t, 'ckpt/{}/student_best.pth'.format(input_compress_tmp))
            state = torch.load(name, map_location='cpu') 
            missing_keys, unexpected_keys = s_model_rank1.load_state_dict(state['state_dict'],strict=False)
            s_model_rank1.cuda()

            input_weight_size = s_model.module.features[0].weight.size(0)
            input_weight_size_rank1 = s_model_rank.module.features[0].weight.size(0)

            select_index = np.arange(input_weight_size-input_compress_tmp)
            print(len(select_index))

            s_model_rank.eval()
            s_model_rank1.eval()

            s_model_dict = s_model.state_dict()
 


            # print(np.sort(acc_list))
            # print(acc_list[select_index])
            # print(select_index.shape)
            # print("aaaaaa")


            s_model_dict_rank = s_model_rank.state_dict()


            for index_i, select_i in enumerate(select_index):
                s_model_dict_rank["module.features.0.weight"][index_i] = \
                s_model_dict["module.features.0.weight"][select_i]


            for i in range(s_model_rank.module.features[3].weight.size(0)):
                for index_j, select_j in enumerate(select_index):
                    s_model_dict_rank["module.features.3.weight"][i][index_j]  = \
                    s_model_dict["module.features.3.weight"][i][select_j] 


            for key in s_model_dict.keys():
                if key in s_model_dict_rank.keys():
                    if (s_model_dict[key].shape == s_model_dict_rank[key].shape):
                        s_model_dict_rank[key] = nn.Parameter(s_model_dict[key].data)


            
            
            missing_keys, unexpected_keys = s_model_rank.load_state_dict(s_model_dict_rank,strict=False)

            s_model_rank.eval()
            s_model_rank1.eval()

                    

            acc_record.reset()
            loss_record.reset()

            for x, target in val_loader:
                x = x[:,0,:,:,:].cuda()
                target = target.cuda()
                with torch.no_grad():
                    output = s_model_rank(x)
                    # print(output)
                    loss = F.cross_entropy(output, target)

                batch_acc = accuracy(output, target, topk=(1,))[0]
                acc_record.update(batch_acc.item(), x.size(0))
                loss_record.update(loss.item(), x.size(0))

            
            acc_random.append(acc_record.avg)
            info = 'cls_acc random :{:.2f}\n'.format(acc_record.avg)
            print(info) 


            acc_record.reset()
            loss_record.reset()

            for x, target in val_loader:
                x = x[:,0,:,:,:].cuda()
                target = target.cuda()
                with torch.no_grad():
                    output = s_model_rank1(x)
                    # print(output)
                    loss = F.cross_entropy(output, target)

                batch_acc = accuracy(output, target, topk=(1,))[0]
                acc_record.update(batch_acc.item(), x.size(0))
                loss_record.update(loss.item(), x.size(0))


            acc_proposion.append(acc_record.avg)
            info = 'cls_acc proposion :{:.2f}\n'.format(acc_record.avg)
            print(info)

            if input_compress_tmp == 62:
                fig = plt.figure()
   
                acc_random = np.array(acc_random)
                acc_proposion = np.array(acc_proposion)
                x_channel_ax = np.array(x_channel_ax)
                ax = fig.add_subplot(111, xlabel=" the number of removing channel", ylabel='accuracy')
                ax.set_title("64 input channel removing test")
                ax.plot(x_channel_ax,acc_random,label="random")
                ax.plot(x_channel_ax,acc_proposion,label="ours")
                ax.legend(loc="lower left")

                fig.savefig('cmp.png')


 


