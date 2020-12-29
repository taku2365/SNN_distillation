#python snn_sskd.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch wrn_16_2 --lr 0.001 --gpu-id 0
# python snn_kd.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch VGG16 --lr 0.0001 --gpu-id "0,1" --epoch 500 --milestones [250,350,400]
# python snn_kd.py --t-path ./experiments/teacher_wrn_40_2_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 250 350 400 --epoch 500  --after_distillation Ture --dataset CIFAR100 
# python snn_kd.py --t-path ./experiments/teacher_ResNet50_seed0_CIFAR100/ --s-arch VGG_SNN_STDB --lr 0.0005 --gpu-id "0,1" --milestones 250 350 400 --epoch 500  --after_distillation Ture --dataset CIFAR100 
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
parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['CIFAR10','CIFAR100'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=str, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


t_name = osp.abspath(args.t_path).split('/')[-1]
t_arch = '_'.join(t_name.split('_')[1:-2])
exp_name = 'sskd_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}+{}_seed{}_{}_{}'.format(\
            args.s_arch, \
            args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight, \
            args.kd_T, args.tf_T, args.ss_T, \
            args.ratio_tf, args.ratio_ss, \
            args.seed, t_name,args.dataset)
exp_path_t = './experiments/{}'.format(exp_name)
os.makedirs(exp_path_t, exist_ok=True)

vgg_after_distillation_CIFAR100= "./experiments/sskd_student_VGG16_weight0.1+0.9+2.7+10.0_T4.0+4.0+0.5_ratio1.0+0.75_seed0_teacher_wrn_40_2_seed0_retrain/ckpt/student_best.pth"


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


# train ssp_head
# for epoch in range(args.t_epoch):



#     t_model.eval()
#     loss_record = AverageMeter()
#     acc_record = AverageMeter()

#     start = time.time()
#     for x, _ in train_loader:

#         t_optimizer.zero_grad()

#         x = x.cuda()
#         c,h,w = x.size()[-3:]
#         x = x.view(-1, c, h, w)

#         a, rep, feat = t_model(x, bb_grad=False)
#         batch = int(x.size(0) / 4)
#         nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
#         aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

#         nor_rep = rep[nor_index]
#         aug_rep = rep[aug_index]
#         nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
#         aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
#         simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
#         target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
#         loss = F.cross_entropy(simi, target)

#         loss.backward()
#         t_optimizer.step()

#         batch_acc = accuracy(simi, target, topk=(1,))[0]
#         loss_record.update(loss.item(), 3*batch)
#         acc_record.update(batch_acc.item(), 3*batch)

#     logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch+1)
#     logger.add_scalar('train/teacher_ssp_acc', acc_record.avg, epoch+1)

#     run_time = time.time() - start
#     info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
#         epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
#     print(info)

#     t_model.eval()
#     acc_record = AverageMeter()
#     loss_record = AverageMeter()
#     start = time.time()
#     for x, _ in val_loader:


#         x = x.cuda()
#         c,h,w = x.size()[-3:]
#         x = x.view(-1, c, h, w)

#         with torch.no_grad():
#             _, rep, feat = t_model(x)
#         print("x {}\n".format(x.shape))
#         batch = int(x.size(0) / 4)
#         print("x.size(0)/4 {}\n".format(x.size(0)/4))
#         print("x1 {}\n".format(x.shape))
#         print("rep {}\n".format(rep.shape))
#         print("feat {}\n".format(feat.shape))

#         nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
#         aug_index = (torch.arange(4*batch) % 4 != 0).cuda()
#         nor_rep = rep[nor_index]
#         aug_rep = rep[aug_index]
#         print("nor {}\n".format(nor_rep.shape))
#         print("aug {}\n".format(aug_rep.shape))
#         print("nor_rep.unsqueeze(2)  {}".format(nor_rep.unsqueeze(2).shape))
#         print("nor_rep.unsqueeze(2).expand(-1,-1,3*batch)  {}".format(nor_rep.unsqueeze(2).expand(-1,-1,3*batch).shape))
#         print("nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transupose(0,2)  {}".format(nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2).shape))
#         nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
#         aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
#         print("nor1 {}\n".format(nor_rep.shape))
#         print("aug1 {}\n".format(aug_rep.shape))
#         simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
#         print("simi {}\n".format(simi.shape))
#         target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
#         print("target {}\n".format(target.shape))
#         loss = F.cross_entropy(simi, target)

#         batch_acc = accuracy(simi, target, topk=(1,))[0]
#         acc_record.update(batch_acc.item(),3*batch)
#         loss_record.update(loss.item(), 3*batch)

#     run_time = time.time() - start
#     logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch+1)
#     logger.add_scalar('val/teacher_ssp_acc', acc_record.avg, epoch+1)

#     info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
#             epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
#     print(info)

#     t_scheduler.step()


# name = osp.join(exp_path_t, 'ckpt/teacher.pth')
# os.makedirs(osp.dirname(name), exist_ok=True)
# torch.save(t_model.state_dict(), name)

# s_model = RESNET_SNN_BATCH_NORM(resnet_name = "RESNET20_BATCH_NORM",labels=100, timesteps=5,dropout=0.3, dataset="CIFAR100",t_divede=5)
if args.s_arch == "VGG_SNN_STDB":
    # if args.retrain:
    #     state = torch.load("{}/ckpt/student_best.pth".format(exp_path_t), map_location='cpu') 
    s_model = VGG_SNN_STDB(vgg_name = "VGG16",labels=num_classes, timesteps=5,dropout=0.1,dataset=args.dataset)
    s_model = nn.DataParallel(s_model)
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

        # s_nor_feat = s_feat[nor_index]
        # s_aug_feat = s_feat[aug_index]
        # s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        # s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        # s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

        # t_nor_feat = t_feat[nor_index]
        # t_aug_feat = t_feat[aug_index]
        # t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        # t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        # t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

        # t_simi = t_simi.detach()
        # aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        # rank = torch.argsort(t_simi, dim=1, descending=True)
        # rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        # index = torch.argsort(rank)
        # tmp = torch.nonzero(rank, as_tuple=True)[0]
        # wrong_num = tmp.numel()
        # correct_num = 3*batch - wrong_num
        # wrong_keep = int(wrong_num * args.ratio_ss)
        # index = index[:correct_num+wrong_keep]
        # distill_index_ss = torch.sort(index)[0]

        # log_simi = F.log_softmax(s_simi / args.ss_T, dim=1)
        # simi_knowledge = F.softmax(t_simi / args.ss_T, dim=1)

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

