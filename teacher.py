# python3 teacher.py --dataset "CIFAR100" --arch wrn_40_2    --lr 0.1  --gpu-id "0,1" --epoch 300 --milestones 60 120 160 --gamma 0.2 --batch-size 128
# python3 teacher.py --dataset "CIFAR100" --arch ResNet50    --lr 0.1  --gpu-id "0,1" --epoch 300 --milestones 60 120 160 --gamma 0.2 --batch-size 128
# python3 teacher.py --dataset "CIFAR10" --arch vgg16_hrank    --lr 0.1  --gpu-id "0,1" --epoch 300 --milestones 120 160 229 --gamma 0.1 --batch-size 128

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
from torchvision.datasets import CIFAR100,CIFAR10
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from self_models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train teacher network.')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[120,180,240])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--arch', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=str, default=0)
parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['CIFAR10','CIFAR100'])





args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

exp_name = 'teacher_{}_seed{}_{}'.format(args.arch, args.seed,args.dataset)
exp_path = './experiments/{}'.format(exp_name)
os.makedirs(exp_path, exist_ok=True)

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
# ])


if args.dataset == 'CIFAR100':
    normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    num_classes      = 100 
elif args.dataset == 'CIFAR10':
    normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    num_classes      = 10


transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
normalize
])
transform_test = transforms.Compose([transforms.ToTensor(), normalize])

if args.dataset == "CIFAR100":
    trainset = CIFAR100('./data', train=True, transform=transform_train,download=True)
    valset = CIFAR100('./data', train=False, transform=transform_test,download=True)
else:
    trainset = CIFAR10('./data', train=True, transform=transform_train,download=True)
    valset = CIFAR10('./data', train=False, transform=transform_test,download=True)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)


model = model_dict[args.arch](num_classes=num_classes).cuda() if args.arch != "vgg16_hrank" else model_dict[args.arch](compress_rate=[0.]*100).cuda() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.Adam(model.parameters(),
#                          lr=args.lr,
#                          amsgrad=True, weight_decay=0, betas=(0.9,0.999))
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

logger = SummaryWriter(osp.join(exp_path, 'events'))
best_acc = -1
for epoch in range(args.epoch):

    model.train()
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    
    start = time.time()
    for x, target in train_loader:

        optimizer.zero_grad()
        x = x.cuda()
        target = target.cuda()

        output = model(x)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        batch_acc = accuracy(output, target, topk=(1,))[0]
        loss_record.update(loss.item(), x.size(0))
        acc_record.update(batch_acc.item(), x.size(0))

    logger.add_scalar('train/cls_loss', loss_record.avg, epoch+1)
    logger.add_scalar('train/cls_acc', acc_record.avg, epoch+1)

    run_time = time.time() - start

    info = 'train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\t'.format(
        epoch+1, args.epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)

    model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:

        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(x)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        loss_record.update(loss.item(), x.size(0))
        acc_record.update(batch_acc.item(), x.size(0))

    run_time = time.time() - start

    logger.add_scalar('val/cls_loss', loss_record.avg, epoch+1)
    logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

    info = 'test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
            epoch+1, args.epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)
    
    scheduler.step()

    # save checkpoint
    if (epoch+1) in args.milestones or epoch+1==args.epoch or (epoch+1)%args.save_interval==0:
        state_dict = dict(epoch=epoch+1, state_dict=model.state_dict(), acc=acc_record.avg)
        name = osp.join(exp_path, 'ckpt/{:03d}.pth'.format(epoch+1))
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)

    # save best
    if acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch+1, state_dict=model.state_dict(), acc=acc_record.avg)
        name = osp.join(exp_path, 'ckpt/best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = acc_record.avg

print('best_acc: {:.2f}'.format(best_acc))
