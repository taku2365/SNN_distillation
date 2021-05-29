#python snn_sskd.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch wrn_16_2 --lr 0.001 --gpu-id 0
# python snn_reduce_channel.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch VGG16 --lr 0.0001 --gpu-id "0,1" --epoch 500 --milestones [250,350,400]
# python snn_reduce_channel.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation Ture --dataset CIFAR100 --timesteps 5 
# python snn_reduce_channel.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation Ture --dataset CIFAR100 --timesteps 5 --input_compress_num 1 --input_compress_rate 0.2 --rank_reduce True
# python snn_reduce_channel.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.00005 --gpu-id "0,1" --milestones 5 10 50 --epoch 15  --after_distillation Ture --dataset CIFAR100 --timesteps 5 --input_compress_num 1 --input_compress_rate 0.2 --rank_reduce True  
# python snn_reduce_channel.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30  --epoch 100  --after_distillation Ture --dataset CIFAR100 
# python snn_reduce_channel.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR10/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 250 350 400 --epoch 500   --dataset CIFAR10 
# sskd_student_VGG_SNN_STDB_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_wrn_40_2_seed0 74.63%  t wideresnet40_2   VGG after_distillation 75~
# \\arch\htdocs\Publications\2021\all-mtg\LT\submission
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
from flops.ptflops import *
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
# print(exp_path_t)
os.makedirs(exp_path_t, exist_ok=True)
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

logger = SummaryWriter(osp.join(exp_path_t, 'events'))

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
    # optimizer =  optim.Adam(s_model.parameters(),
    #                 lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))        
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
            # print("E rate",E_ANN/E_SNN)
            
            # print(output)
            # print(spike_rate)
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

    s_model.module.cal_neuron = False


    


 
    
    

    if args.rank_reduce:
        # s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset)
        state = torch.load(vgg_stdb_after_distillation_CIFAR100, map_location='cpu') 
        optimizer =  optim.Adam(s_model.parameters(),
                        lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))
        missing_keys, unexpected_keys = s_model.load_state_dict(state['state_dict'],strict=False)
        s_model.module.rank_reduce = False
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        s_model.cuda()
        s_model.eval()
        # optimizer =  optim.Adam(s_model.parameters(),
        #                 lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))        
        s_model_rank.cuda()
        s_model_rank.eval()
        s_model_rank1.cuda()
        s_model_rank1.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()




        s_model_dict = s_model.state_dict()
        s_model_dict_rank = s_model_rank.state_dict()



    

        input_compress_tmp = 0


        for _ in range(s_model.module.features[0].weight.size(0)):

            input_compress_tmp += 2


                

            if (input_compress_tmp != 2):
                s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=input_compress_tmp-2)
                s_model = nn.DataParallel(s_model)
                name = osp.join(exp_path_t, 'ckpt/{}/student_best.pth'.format(input_compress_tmp-2))
                state = torch.load(name, map_location='cpu') 
                missing_keys, unexpected_keys = s_model.load_state_dict(state['state_dict'],strict=False)

                s_model_rank = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=input_compress_tmp-1)
                s_model_rank = nn.DataParallel(s_model_rank)    
                s_model_rank1 = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_num=input_compress_tmp)
                s_model_rank1 = nn.DataParallel(s_model_rank1)

                print("Bbbbb")

                print(s_model.module.features[0].weight.size(0))


            
                if(osp.join(exp_path_t, 'ckpt/{}/student_best.pth'.format(input_compress_tmp-2))):
                    s_model_rank1.eval()
                    acc_record.reset()
                    loss_record.reset()
                    print(input_compress_tmp-2)
                    s_model.cuda()
                    s_model.eval()
    
                    for x, target in val_loader:
                        x = x[:,0,:,:,:].cuda()
                        target = target.cuda()
                        with torch.no_grad():
                            output = s_model(x)
                            # print(output)
                            loss = F.cross_entropy(output, target)

                        batch_acc = accuracy(output, target, topk=(1,))[0]
                        acc_record.update(batch_acc.item(), x.size(0))
                        loss_record.update(loss.item(), x.size(0))


                    info = 'cls_acc:{:.2f}\n'.format(acc_record.avg)
                    print(info)

                    continue

            


            if not os.path.exists('./experiments/'+args.s_arch+args.s_arch+"_"+str(input_compress_tmp)+"_"+'conv_input_importance.npy'):

                acc_list = []
                acc_record = AverageMeter() 
                loss_record = AverageMeter()

                for del_index in range(s_model.module.features[0].weight.size(0)):
            

                    s_model_index_input = np.arange(s_model.module.features[0].weight.size(0))
                    s_model_index_input = np.delete(s_model_index_input,del_index)



                    for index_i, select_i in enumerate(s_model_index_input):
                        s_model_dict_rank["module.features.0.weight"][index_i] = \
                        s_model_dict["module.features.0.weight"][select_i]


                    for i in range(s_model_rank.module.features[3].weight.size(0)):
                        for index_j, select_j in enumerate(s_model_index_input):
                            s_model_dict_rank["module.features.3.weight"][i][index_j]  = \
                            s_model_dict["module.features.3.weight"][i][select_j] 


                    for key in s_model_dict.keys():
                        if key in s_model_dict_rank.keys():
                            if (s_model_dict[key].shape == s_model_dict_rank[key].shape):
                                s_model_dict_rank[key] = nn.Parameter(s_model_dict[key].data)
                            

                    missing_keys, unexpected_keys = s_model_rank.load_state_dict(s_model_dict_rank,strict=False)
                    # print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
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
                    



                    info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
                    acc_list.append(acc_record.avg)
                    print(info)
                    print(np.sort(acc_list))
                    print(np.argsort(acc_list))
                
                acc_list_sort = np.argsort(acc_list)
                np.save('./experiments/'+args.s_arch+args.s_arch+"_"+str(input_compress_tmp)+"_"+'conv_input_importance.npy', acc_list_sort)
                np.save('./experiments/'+args.s_arch+args.s_arch+"_"+str(input_compress_tmp)+"_"+'conv_input_importance_acc.npy', acc_list)


            if(not os.path.exists(osp.join(exp_path_t, 'ckpt/{}/student_best.pth'.format(input_compress_tmp)))):
    

                
                acc_list = np.load('./experiments/'+args.s_arch+args.s_arch+"_"+str(input_compress_tmp)+"_"+'conv_input_importance_acc.npy')
                input_weight_size = s_model.module.features[0].weight.size(0)
                input_weight_size_rank1 = s_model_rank1.module.features[0].weight.size(0)

                select_index = np.argsort(acc_list)[:input_weight_size_rank1]
                print(np.sort(acc_list))
                print(acc_list[select_index])
                print(select_index.shape)
                print("aaaaaa")
                select_index.sort()

                s_model_dict_rank1 = s_model_rank1.state_dict()


                for index_i, select_i in enumerate(select_index):
                    s_model_dict_rank1["module.features.0.weight"][index_i] = \
                    s_model_dict["module.features.0.weight"][select_i]


                for i in range(s_model_rank1.module.features[3].weight.size(0)):
                    for index_j, select_j in enumerate(select_index):
                        s_model_dict_rank1["module.features.3.weight"][i][index_j]  = \
                        s_model_dict["module.features.3.weight"][i][select_j] 


                for key in s_model_dict.keys():
                    if key in s_model_dict_rank.keys():
                        if (s_model_dict[key].shape == s_model_dict_rank1[key].shape):
                            s_model_dict_rank1[key] = nn.Parameter(s_model_dict[key].data)
                        


                

                missing_keys, unexpected_keys = s_model_rank1.load_state_dict(s_model_dict_rank1,strict=False)

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


                info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
                print(info) 

                best_acc = 0

                acc_record.reset()
                loss_record.reset()

 


                for epoch in range(args.epoch):

                    # train
                    s_model_rank1.train()
                    loss1_record = AverageMeter()
                    loss2_record = AverageMeter()
                    loss3_record = AverageMeter()
                    loss4_record = AverageMeter()
                    cls_acc_record = AverageMeter()
                    ssp_acc_record = AverageMeter()

                    start = time.time()
                    for x, target in train_loader:
                        optimizer.zero_grad()

                        c,h,w = x.size()[-3:]
                        x = x.view(-1,c,h,w).cuda()
                        # print(x.size())
                        target = target.cuda()

                        batch = int(x.size(0) / 4)
                        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
                        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()
                        # print(x.shape)
                        output = s_model_rank1(x)

                        # print(output)

                        
                        log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)
                        log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1)
                        # print(log_aug_output)
                        with torch.no_grad():
                            knowledge = t_model(x)
                            nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
                            aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)

                        # error level ranking
                        aug_target = target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
                        rank = torch.argsort(aug_knowledge, dim=1, descending=True)
                        rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
                        index = torch.argsort(rank)
                        tmp = torch.nonzero(rank, as_tuple=True)[0]
                        wrong_num = tmp.numel()
                        correct_num = 3*batch - wrong_num
                        wrong_keep = int(wrong_num * args.ratio_tf)
                        index = index[:correct_num+wrong_keep]
                        distill_index_tf = torch.sort(index)[0]


                        loss1 = F.cross_entropy(output[nor_index], target)
                        loss2 = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * args.kd_T * args.kd_T
                        loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf], \
                                        reduction='batchmean') * args.tf_T * args.tf_T
                        # loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                        #                 reduction='batchmean') * args.ss_T * args.ss_T

                        loss = args.ce_weight * loss1 + args.kd_weight * loss2 + args.tf_weight * loss3 
                        # loss = args.ce_weight * loss1 + args.kd_weight * loss2 + args.tf_weight * loss3 + args.ss_weight * loss4

                        loss.backward()
                        optimizer.step()

                        cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
                        # ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
                        loss1_record.update(loss1.item(), batch)
                        loss2_record.update(loss2.item(), batch)
                        loss3_record.update(loss3.item(), len(distill_index_tf))
                        # loss4_record.update(loss4.item(), len(distill_index_ss))
                        cls_acc_record.update(cls_batch_acc.item(), batch)
                        # ssp_acc_record.update(ssp_batch_acc.item(), 3*batch)
                        # for key, value in s_model_rank1.module.leak.items():
                        #         # maximum of leak=1.0
                        #     s_model_rank1.module.leak[key].data.clamp_(max=1.0)

                    logger.add_scalar('train/ce_loss', loss1_record.avg, epoch+1)
                    logger.add_scalar('train/kd_loss', loss2_record.avg, epoch+1)
                    logger.add_scalar('train/tf_loss', loss3_record.avg, epoch+1)
                    # logger.add_scalar('train/ss_loss', loss4_record.avg, epoch+1)
                    logger.add_scalar('train/cls_acc', cls_acc_record.avg, epoch+1)
                    # logger.add_scalar('train/ss_acc', ssp_acc_record.avg, epoch+1)

                    run_time = time.time() - start
                    info = 'student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(
                        epoch+1, args.epoch, run_time, loss1_record.avg, loss2_record.avg, cls_acc_record.avg)
                    print(info)

                    # cls val
                    s_model_rank1.eval()
                    acc_record = AverageMeter()
                    loss_record = AverageMeter()
                    start = time.time()
                    for x, target in val_loader:

                        x = x[:,0,:,:,:].cuda()
                        target = target.cuda()
                        with torch.no_grad():
                            output  = s_model_rank1(x)
                            loss = F.cross_entropy(output, target)

                        batch_acc = accuracy(output, target, topk=(1,))[0]
                        acc_record.update(batch_acc.item(), x.size(0))
                        loss_record.update(loss.item(), x.size(0))

                    run_time = time.time() - start
                    logger.add_scalar('val/ce_loss', loss_record.avg, epoch+1)
                    logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

                    info = 'student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_acc:{:.2f}\n'.format(
                            epoch+1, args.epoch, run_time, acc_record.avg)

                    print(info)

                    if acc_record.avg > best_acc:
                        best_acc = acc_record.avg
                        state_dict = dict(epoch=epoch+1, state_dict=s_model_rank1.state_dict(), best_acc=best_acc)
                        name = osp.join(exp_path_t, 'ckpt/{}/student_best.pth'.format(input_compress_tmp))
                        os.makedirs(osp.dirname(name), exist_ok=True)
                        torch.save(state_dict, name)

                    scheduler.step()



