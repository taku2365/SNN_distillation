#python snn_sskd.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch wrn_16_2 --lr 0.001 --gpu-id 0
# python snn_kd.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch VGG16 --lr 0.0001 --gpu-id "0,1" --epoch 500 --milestones [250,350,400]
# python snn_kd.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation Ture --dataset CIFAR100 --timesteps 5 
# python snn_kd.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30 50 --epoch 70  --after_distillation Ture --dataset CIFAR100 --timesteps 5 --input_compress_rate 0.45 --rank_reduce True
# python snn_kd.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 15 30  --epoch 100  --after_distillation Ture --dataset CIFAR100 
# python snn_kd.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR10/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 250 350 400 --epoch 500   --dataset CIFAR10 
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
parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])
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
parser.add_argument('--input_compress_rate', default=0,   type=float )
parser.add_argument('--rank_reduce', default=False,   type=bool )


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

def get_feature_hook(self, input, output):
    global feature_result
    # global entropy
    global total
    global num
    a = output.shape[0]
    b = output.shape[1]
    # print(output.size())
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
    # print(c)
    # print(num)
    num += 1
    # print(output.size())
    # print(output[0,0,:,:].size())
    # print(torch.matrix_rank(output[0,0,:,:]).size())
    # print(torch.matrix_rank(output[0,0,:,:]))
    c = c
    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

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

for x, target in val_loader:

    
    x = x[:,0,:,:,:].cuda()
    target = target.cuda()
    with torch.no_grad():
        output = t_model(x)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    acc_record.update(batch_acc.item(), x.size(0))
    loss_record.update(loss.item(), x.size(0))


info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
print(info)





net_name = "RESNET20_BATCH_NORM",labels=100, timesteps=5,dropout=0.3, dataset="CIFAR100",t_divede=5)
if args.s_arch == "VGG_SNN_STDB":
    # if args.retrain:
    #     state = torch.load("{}/ckpt/student_best.pth".format(exp_path_t), map_location='cpu') 
    s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,rank_reduce=args.rank_reduce)
    s_model = nn.DataParallel(s_model)
    s_model_rank = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=args.timesteps,dropout=0.1,dataset=args.dataset,input_compress_rate=args.input_compress_rate)
    s_model_rank = nn.DataParallel(s_model_rank)
    # s_model.cuda()
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
    # if args.self_distillation:
    #     t_model = VGG(vgg_name="VGG16", labels=100, dataset="CIFAR100")
    #     t_model = nn.DataParallel(t_model)
    #     state   = torch.load(vgg_after_distillation,map_location='cpu')
    #     missing_keys, unexpected_keys = t_model.load_state_dict(state['state_dict'], strict=False)
    #     t_model.cuda()
    #     t_model.eval()
    missing_keys, unexpected_keys = s_model.load_state_dict(state['state_dict'],strict=False)
    print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))

    s_model.module.threshold_update(scaling_factor = 0.6, thresholds=thresholds[:])
    s_model.cuda()
    optimizer =  optim.Adam(s_model.parameters(),
                        lr=args.lr, amsgrad=True, weight_decay=0, betas=(0.9,0.999))
    
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
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        # for x, target in val_loader:

        #     x = x[:,0,:,:,:].cuda()
        #     target = target.cuda()
        #     with torch.no_grad():
        #         output = s_model(x)
        #         loss = F.cross_entropy(output, target)

        #     batch_acc = accuracy(output, target, topk=(1,))[0]
        #     acc_record.update(batch_acc.item(), x.size(0))
        #     loss_record.update(loss.item(), x.size(0))
        # info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
        # print(info)
        # exit(1)
        # handler = s_model.module.features[0].register_forward_hook(get_feature_hook)


        if not os.path.exists('./experiments/'+args.s_arch+'_rank_conv_input.npy'):

            acc_record = AverageMeter()
            loss_record = AverageMeter()
            feature_result = 0
            total = 0
            i = 0

            for x, target in train_loader:
                x = x[:,0,:,:,:].cuda()
                target = target.cuda()
                with torch.no_grad():
                    output,input_rank_sum = s_model(x)
                    print(i)
                    i += 1
                    # print(input_rank_sum.size())
                    # print(input_rank)

                    # a = input_rank.shape[0]
                    # b = input_rank.shape[1]
                    # c = torch.tensor([torch.matrix_rank(input_rank[i,j,:,:]/args.timesteps).cuda().item() for i in range(a) for j in range(b)]).cuda()
                    # c = c.view(a, -1).float()
                    # c = c.sum(0)
                    feature_result = feature_result * total + input_rank_sum
                    total = total + input_rank_sum.size(0)
                    feature_result = feature_result / total
                    # print(1,feature_result[0:20])
                    # print(2,feature_result[20:40])
                    # print(3,feature_result[40:63])
                    # loss = F.cross_entropy(output, target)

                batch_acc = accuracy(output, target, topk=(1,))[0]
                acc_record.update(batch_acc.item(), x.size(0))
                # loss_record.update(loss.item(), x.size(0))


            np.save('./experiments/'+args.s_arch+'_rank_conv_input.npy', feature_result.cpu().numpy())
        
        else:
            feature_result = np.load('./experiments/'+args.s_arch+'_rank_conv_input.npy')
        
        input_weight_size = s_model.module.features[0].weight.size(0)
        input_weight_size_rank = s_model_rank.module.features[0].weight.size(0)
        select_index = np.argsort(feature_result)[input_weight_size-input_weight_size_rank:]
        print(feature_result[select_index]) 
        print(select_index)
        select_index.sort()
        print(select_index)
        s_model_dict = s_model.state_dict()
        s_model_dict_rank = s_model_rank.state_dict()
        # s_model_index_input = np.arange(input_weight_size)
        # s_model_index_input = s_model_index_input[input_weight_size-input_weight_size_rank:]


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
                
                else:
                    print(key)


        

        missing_keys, unexpected_keys = s_model_rank.load_state_dict(s_model_dict_rank,strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        acc_record = AverageMeter()
        loss_record = AverageMeter()
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
        print(info)
        exit(1)


    # # print(s_model_rank.module.features[0].weight.size())
    # # print(s_model.state_dict()["module.features.0.weight"])
    # s_model_dict = s_model.state_dict()
    # s_model_dict_rank = s_model_rank.state_dict()

    # input_weight_size = s_model.module.features[0].weight.size(0)
    # input_weight_size_rank = s_model_rank.module.features[0].weight.size(0)
    # s_model_index_input = np.arange(input_weight_size)
    # s_model_index_input = s_model_index_input[input_weight_size-input_weight_size_rank:]
    # # print(s_model_rank.module.features[0].weight.size(0))
    # # print(input_weight_size-input_weight_size_rank)
    # # print(s_model_index_input)
    # # print(s_model_dict_rank["module.features.3.weight"][0][0])
    # # print(s_model_dict_rank["module.features.0.weight"][0][0])
    # print(s_model_dict_rank["module.features.3.weight"].size())



    # for index_i, i in enumerate(s_model_index_input):
    #         s_model_dict_rank["module.features.0.weight"][index_i] = \
    #         s_model_dict["module.features.0.weight"][i]


    # for i in range(s_model_rank.module.features[3].weight.size(0)):
    #     for index_j, j in enumerate(s_model_index_input):
    #         s_model_dict_rank["module.features.3.weight"][i][index_j]  = \
    #         s_model_dict["module.features.3.weight"][i][j] 
    
    # # print(s_model_dict["module.features.0.weight"][29][0])
    # # print(s_model_dict_rank["module.features.0.weight"][0][0])
    # # print(s_model_dict["module.features.3.weight"][0][29])
    # # print(s_model_dict_rank["module.features.3.weight"][0][0])
    # # print(s_model_dict_rank["module.features.3.weight"][0].size())
    # # print(s_model_dict_rank["module.features.3.weight"].size())
    # exit(1)
    
    
elif args.s_arch == "VGG16":
    s_model = VGG(vgg_name="VGG16", labels=100, dataset=args.dataset)
    s_model = nn.DataParallel(s_model)
    if args.dataset == "CIFAR100":
        if args.retrain:
            state = torch.load("{}/ckpt/student_best.pth".format(exp_path_t), map_location='cpu')  
        else:
            state = torch.load("trained_models/ann_vgg16_cifar10_best_model.pth", map_location='cpu')     
 
    if args.dataset == "CIFAR100":
        cur_dict = s_model.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):

                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
        missing_keys, unexpected_keys = s_model.load_state_dict(cur_dict,strict=False)

    s_model.cuda()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    s_model = model_dict[args.s_arch](num_classes=100)
    s_model = nn.DataParallel(s_model)
    s_model.cuda()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    






# print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
# print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
# s_model = wrapper(module=s_model).cuda()
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)


best_acc = 0
for epoch in range(args.epoch):

    # train
    s_model.train()
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
        output = s_model(x)

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
        # for key, value in s_model.module.leak.items():
        #         # maximum of leak=1.0
        #     s_model.module.leak[key].data.clamp_(max=1.0)

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
    s_model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:

        x = x[:,0,:,:,:].cuda()
        target = target.cuda()
        with torch.no_grad():
            output  = s_model(x)
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
        state_dict = dict(epoch=epoch+1, state_dict=s_model.state_dict(), best_acc=best_acc)
        name = osp.join(exp_path_t, 'ckpt/student_best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)

    scheduler.step()

