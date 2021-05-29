# python spike_rate_visualize.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.00005 --gpu-id "0" --milestones 5 10 50 --epoch 15  --after_distillation Ture --dataset CIFAR100 --timesteps 5   
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

from utils import AverageMeter, accuracy
# from wrapper import wrapper
from cifar import CIFAR100_aug,CIFAR10_aug
from torchvision.datasets import CIFAR100,CIFAR10
from self_models import *
from self_models import model_dict

import matplotlib.pyplot as plt

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
parser.add_argument('--vgg_after_distillation',type=str )
parser.add_argument('--vgg_stdb_after_distillation',type=str )

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


t_name = osp.abspath(args.t_path).split('/')[-1]
t_arch = '_'.join(t_name.split('_')[1:-2])

linear_flops = []
conv_flops = []
vgg_after_distillation_CIFAR100= args.vgg_after_distillation
vgg_stdb_after_distillation_CIFAR100= args.vgg_stdb_after_distillation

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



acc_record = AverageMeter()
loss_record = AverageMeter()
start = time.time()





def linear_flops_counter_hook(module, input, output):
    global linear_flops 
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    # module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)
    linear_flops.append(int(np.prod(input.shape) * output_last_dim + bias_flops))

def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    global conv_flops
    input = input[0]
    # print(conv_module)
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    # conv_module.__flops__ += int(overall_flops)
    # print(conv_module,int(overall_flops),conv_per_position_flops,active_elements_count,output_dims,output.shape)
    conv_flops.append(int(overall_flops))


# net_name = "RESNET20_BATCH_NORM",labels=100, timesteps=5,dropout=0.3, dataset="CIFAR100",t_divede=5)
if args.s_arch == "VGG_SNN_STDB":
    # if args.retrain:
    #     state = torch.load("{}/ckpt/student_best.pth".format(exp_path_t), map_location='cpu') 
    s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset)
    op_counter=True
    if(op_counter):
        s_model.to("cuda")
        hook_handles = []

        for layer in s_model.modules():
            
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(conv_flops_counter_hook)
                hook_handles.append(handle)

            elif isinstance(layer,nn.Linear):
                handle = layer.register_forward_hook(linear_flops_counter_hook)
                hook_handles.append(handle)

        
        dsize = (1, 3, 32, 32)
        inputs = torch.randn(dsize).to("cuda")
        s_model(inputs)

        
        # print(linear_flops[:int(len(linear_flops)/args.timesteps)])
        # print("conv flops len",len(conv_flops[:int(len(conv_flops)/args.timesteps)]))

        conv_flops = conv_flops[:int(len(conv_flops)/args.timesteps)]
        linear_flops = linear_flops[:int(len(linear_flops)/args.timesteps)-1]
        

            

        # with torch.cuda.device(0):
            
        #     macs, params = get_model_complexity_info(s_model, (3, 32, 32), as_strings=True,
        #                                     print_per_layer_stat=True, verbose=True)
        
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        # exit(1)
    s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,cal_neuron=True)
    s_model = nn.DataParallel(s_model)    
    s_model_rank = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=1)
    s_model_rank = nn.DataParallel(s_model_rank)    
    s_model_rank1 = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=2)
    s_model_rank1 = nn.DataParallel(s_model_rank1)
    print(repr(s_model))



    s_model.cuda()
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
    optimizer =  optim.Adam(s_model.parameters(),
                        lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))

    
    acc_record.reset()
    loss_record.reset()
    s_model.eval()


    state = torch.load(vgg_stdb_after_distillation_CIFAR100, map_location='cpu') 
    optimizer =  optim.Adam(s_model.parameters(),
                    lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))
    missing_keys, unexpected_keys = s_model.load_state_dict(state['state_dict'],strict=False)
    s_model.module.rank_reduce = False
    print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
    s_model.cuda()
    s_model.eval()      
    s_model_rank.cuda()
    s_model_rank.eval()
    s_model_rank1.cuda()
    s_model_rank1.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()



    # dsize = (1, 3, 224, 224)

    # inputs = torch.randn(dsize)
    # total_ops, total_params = profile(s_model, (inputs,), verbose=False)
    # print("%s | %.2f | %.2f" % ("VGG_SNN", total_params / (1000 ** 2), total_ops / (1000 ** 3)))

    s_model_dict = s_model.state_dict()
    s_model_dict_rank = s_model_rank.state_dict()
    tmp_len = 0
    E_ANN = 0
    E_SNN = 0
    flops1 = np.concatenate((conv_flops,linear_flops))


    for x, target in val_loader:
        tmp_len += 1 
        x = x[:,0,:,:,:].cuda()
        target = target.cuda()
        with torch.no_grad():
            output,spike_rate_conv,spike_rate_linear = s_model(x)
            spike_rate = np.concatenate((spike_rate_conv,spike_rate_linear))
  
            FLOPs_conv_ANN = np.sum(conv_flops)
            FLOPs_conv_SNN = np.sum(conv_flops*spike_rate_conv)
            FLOPs_linear_ANN = np.sum(linear_flops)
            FLOPs_linear_SNN = np.sum(linear_flops*spike_rate_linear)
            E_ANN += (FLOPs_conv_ANN+FLOPs_linear_ANN)*4.6
            E_SNN += (FLOPs_conv_SNN+FLOPs_linear_SNN)*0.9
            loss = F.cross_entropy(output, target)

    
    print(np.mean(spike_rate))
    print(np.sum(flops1*spike_rate)/np.sum(flops1))
    print(E_ANN/E_SNN)

    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))#y軸小数点以下3桁表示
    # plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    left = np.arange(len(spike_rate))
    plt.title("Spike rate for each layer")
    plt.xlabel("Layer [l]")
    plt.ylabel("Spike Rate R(l)")
    plt.bar(left, spike_rate)
    plt.savefig("Spike_Rate.png")

    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))#y軸小数点以下3桁表示
    # plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    plt.title("Conv + linear FLOPs")
    plt.xlabel("Layer [l]")
    plt.ylabel("MFlops")
    ANN_conv_Mflops = []
    SNN_conv_Mflops = []
    left = np.arange(len(flops1))
    for i,flops in enumerate(flops1):
        ANN_conv_Mflops.append(flops/10.**6)
        SNN_conv_Mflops.append(spike_rate[i]*flops/10.**6)

    plt.bar(left, ANN_conv_Mflops,label="ANN")
    plt.bar(left,SNN_conv_Mflops,label="SNN")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
    plt.savefig("conv_flops.png")


    batch_acc = accuracy(output, target, topk=(1,))[0]
    acc_record.update(batch_acc.item(), x.size(0))
    loss_record.update(loss.item(), x.size(0))


    info = 'cls_acc:{:.2f}\n'.format(acc_record.avg)
    print(info)
