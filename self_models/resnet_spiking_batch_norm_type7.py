##### fix stdb bug  
##python3 snn.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.1 --scaling_factor 0.3  --weight_decay 0.0 --pretrained_ann trained_models/ann_resnet20_cifar10_batch_on.pth  --lr_interval '0.10 0.40 0.70' --lr_reduce 5  --thresholds_new True --store_name batch_type3 --lr_interval '0.10 0.40 0.70' --lr_reduce 5 --timesteps 25                      
##python3 snn_only.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.1 --scaling_factor 0.3  --weight_decay 0.0  --store_name batch_type3 --lr_interval '0.15 0.40 0.70' --lr_reduce 5 --timesteps 25  --epoch 500 -lr 0.002              
## python3 snn.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0 --pretrained_snn trained_models/snn_resnet20_batch_norm_cifar10_40_lr0.005_batch_type5_t_divide5.pth --lr_reduce 10 --timesteps 40 -lr 0.005 --store_name batch_type5_retrain --lr_interval '0.10 0.40 0.70' --t_divide 5 --lr_reduce 5 --timesteps 40  --epochs 500  --retrain True
## python3 snn.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0  --lr_reduce 10  --store_name batch_type4_dropout03_epoch150 --lr_interval '0.3 0.60 0.70' --lr_reduce 5 --timesteps 40  --epochs 500 -lr 0.01 --default_threshold 1
#python3 snn.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0  --lr_reduce 10  --store_name batch_type7 --lr_interval '0.3 0.42 0.70' --lr_reduce 10 --timesteps 5  --epochs 500 -lr 0.002 --default_threshold 1 --t_divide 5 --dataset cifar100
#python3 snn.py -a VGG16_STDB --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0  --lr_reduce 10  --store_name batch_type7 --lr_interval '0.3 0.42 0.70' --lr_reduce 10 --timesteps 5  --epochs 500 -lr 0.002 --default_threshold 1 --t_divide 5 --dataset cifar100
#python3 snn.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0  --lr_reduce 10  --store_name batch_type7 --lr_interval '0.3 0.42 0.70' --lr_reduce 10 --timesteps 5  --epochs 500 -lr 0.002 --default_threshold 1 --t_divide 5 

#python3 snn.py -a RESNET20_BATCH_NORM --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0  --lr_reduce 10  --store_name batch_type5 --lr_interval '0.3 0.42 0.70' --lr_reduce 10 --timesteps 5  --epochs 500 -lr 0.002 --default_threshold 1 --t_divide 5 --retrain True --pretrained_snn trained_models/snn_resnet20_batch_norm_cifar10_5_lr0.002_batch_type5_t_divide5.pth

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
import copy



cfg = {
	'resnet6'	            : [1,1,0,0],
	'resnet12' 	            : [1,1,1,1],
	'resnet20'	            : [2,2,2,2],
	'resnet20_batch_norm'	: [2,2,2,2],
	'resnet34'	            : [3,4,6,3]
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class SeparatedBatchNorm1d(nn.Module):

	"""
	A batch normalization module which keeps its running mean
	and variance separately per timestep.
	"""

	def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
				affine=True):
		"""
		Most parts are copied from
		torch.nn.modules.batchnorm._BatchNorm.
		"""

		super(SeparatedBatchNorm1d, self).__init__()
		self.num_features = num_features
		self.max_length = max_length
		self.affine = affine
		self.eps = eps
		self.momentum = momentum
		if self.affine:
			self.weight = nn.Parameter(torch.FloatTensor(num_features))
			self.bias = nn.Parameter(torch.FloatTensor(num_features))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		for i in range(max_length):
			self.register_buffer(
				'running_mean_{}'.format(i), torch.zeros(num_features))
			self.register_buffer(
				'running_var_{}'.format(i), torch.ones(num_features))
		self.reset_parameters()

	def reset_parameters(self):
		for i in range(self.max_length):
			running_mean_i = getattr(self, 'running_mean_{}'.format(i))
			running_var_i = getattr(self, 'running_var_{}'.format(i))
			running_mean_i.zero_()
			running_var_i.fill_(1)
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def _check_input_dim(self, input_):
		if input_.size(1) != self.running_mean_0.nelement():
			raise ValueError('got {}-feature tensor, expected {}'
							.format(input_.size(1), self.num_features))

	def forward(self, input_, time):

		self._check_input_dim(input_)
		if time >= self.max_length:
			time = self.max_length - 1
		running_mean = getattr(self, 'running_mean_{}'.format(time))
		running_var = getattr(self, 'running_var_{}'.format(time))

		return F.batch_norm(
			input=input_, running_mean=running_mean, running_var=running_var,
			weight=self.weight, bias=self.bias, training=self.training,
			momentum=self.momentum, eps=self.eps)

	def __repr__(self):
		return ('{name}({num_features}, eps={eps}, momentum={momentum},'
				' max_length={max_length}, affine={affine})'
				.format(name=self.__class__.__name__, **self.__dict__))



class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout,batch_flag,bias,timesteps):
        #print('In __init__ BasicBlock')
        #super(BasicBlock, self).__init__()
        super().__init__()
        self.identity_conv_flag = False

        
        self.residual = nn.Sequential(
             SeparatedBatchNorm1d(num_features=in_planes, max_length=timesteps),
             nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True),
             nn.Dropout(p=dropout),
             SeparatedBatchNorm1d(num_features=planes, max_length=timesteps),
             nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
            )

        self.identity = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.identity_conv_flag = True
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )




        # self.bn1 = nn.BatchNorm2d(in_planes)
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        # self.dropout = nn.Dropout(p=dropout_rate)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)



    def forward(self, dic):
        
        out_prev 		= dic['out_prev']
        pos 			= dic['pos']
        act_func 		= dic['act_func']
        mem 			= dic['mem']
        mask 			= dic['mask']
        threshold 		= dic['threshold']
        t 				= dic['t']
        t_th 		    = dic['t_th']
        leak			= dic['leak']
        find_max_mem 	= dic['find_max_mem']
        inp				= out_prev.clone()

        #SeparatedBatchNorm1d_1
        residual_batch1  = self.residual[0](inp,t) 

        #liner spike1 
        out 			= act_func(residual_batch1)
        out_prev  		= out.clone() 

        #conv1
        conv1            = self.residual[1](out_prev)
        mem[pos] 		= getattr(leak, 'l'+str(t_th).zfill(2)+'_'+str(pos).zfill(2)) *mem[pos] + conv1
        mem_thr 		= (mem[pos]/getattr(threshold,'t'+str(t_th).zfill(2)+'_' +str(pos).zfill(2))) - 1.0
        rst 			= getattr(threshold,'t'+str(t_th).zfill(2)+'_'+str(pos).zfill(2)) * (mem_thr>0).float()
        mem[pos] 		= mem[pos] - rst


        #dropout1
        out_prev 		= mem_thr * mask[pos]

        #SeparatedBatchNorm1d_2
        residual_batch2  = self.residual[3](out_prev,t)

        #liner spike2
        out 			= act_func(residual_batch2)
        out_prev  		= out.clone() 


        #conv2+identity
        # identity  = self.identity_batch(self.identity(inp),t) if self.identity_conv_flag else self.identity(inp)
        identity        = self.identity(inp)
        residual_batch  = self.residual[4](out_prev) + identity
        mem[pos+1]     = getattr(leak, 'l'+str(t_th).zfill(2)+'_'+str(pos+1).zfill(2)) *mem[pos+1]+residual_batch
        mem_thr 		= (mem[pos+1]/getattr(threshold, 't'+str(t_th).zfill(2)+'_'+str(pos+1).zfill(2))) - 1.0
        rst 			= getattr(threshold, 't'+str(t_th).zfill(2)+'_'+str(pos+1).zfill(2)) * (mem_thr>0).float()
        mem[pos+1] 		= mem[pos+1] - rst


        return mem_thr

class RESNET_SNN_BATCH_NORM(nn.Module):
	
    #all_layers = []
    #drop 		= 0.2
    def __init__(self, resnet_name, activation='Linear', labels=10, timesteps=75, leak=1.0, default_threshold=1, dropout=0.2, dataset='CIFAR10',batch_flag=False,bias=False,t_divede=5):

        super().__init__()
        
        self.resnet_name	         = resnet_name.lower()
        self.act_func 		         = LinearSpike.apply
        self.labels 		         = labels
        self.timesteps 		         = timesteps
        self.dropout 		         = dropout
        self.dataset 		         = dataset
        self.mem 			         = {}
        self.mask 			         = {}
        # self.spike 			         = {}
        self.pre_process_batch_norm  = {}
        self.t_divide                = t_divede
        self.labels                  = labels

        depth = 28
        widen_factor = 20
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]
        self.n = (depth-4)/6
        self.k = widen_factor


        

                    


        block               = BasicBlock
        self.in_planes      = 16

        #conv1
        
        self.conv1 = conv3x3(3,nStages[0])
                                    


        self.layer1 		= self._make_layer(block,nStages[1],int(self.n), stride=1, dropout=self.dropout\
            ,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
        self.layer2 		= self._make_layer(block,nStages[2],int(self.n), stride=2, dropout=self.dropout\
            ,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
        self.layer3 		= self._make_layer(block,nStages[3],int(self.n), stride=2, dropout=self.dropout\
            ,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
        #self.avgpool 		= nn.AvgPool2d(2)

        self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3}
  
        #batch
        self.bn = SeparatedBatchNorm1d(num_features=nStages[3], max_length=self.timesteps)
        
        #linear
        self.linear = nn.Linear(nStages[3], labels)




        self._initialize_weights2()

        threshold 	= {}
        lk 			= {}

        for t in range(self.timesteps):

            t = int(t//self.t_divide)
            print(t)

            #conv1
            pos = 0
            threshold['t'+str(t).zfill(2)+'_'+str(pos).zfill(2)] 	= nn.Parameter(torch.tensor(default_threshold))
            lk['l'+str(t).zfill(2)+'_'+str(pos).zfill(2)] 	  		= nn.Parameter(torch.tensor(leak))

            pos += 1

                    
            for i in range(1,4):

                layer = self.layers[i]
                for index in range(len(layer)):
                    for l in range(len(layer[index].residual)):
                        if isinstance(layer[index].residual[l],nn.Conv2d):
                            # print('t'+str(t).zfill(2)+'_'+str(pos).zfill(2))
                            threshold['t'+str(t).zfill(2)+'_'+str(pos).zfill(2)] = nn.Parameter(torch.tensor(default_threshold))
                            lk['l'+str(t).zfill(2)+'_'+str(pos).zfill(2)] 		= nn.Parameter(torch.tensor(leak))
                            pos=pos+1
            

            threshold['t'+str(t).zfill(2)+'_'+str(pos).zfill(2)] 		= nn.Parameter(torch.tensor(default_threshold))
            lk['l'+str(t).zfill(2)+'_'+str(pos).zfill(2)] 				= nn.Parameter(torch.tensor(leak)) 



                    
            self.threshold 	= nn.ParameterDict(threshold)
            self.leak 		= nn.ParameterDict(lk)


        
        threshold_length = 0

        for key, value in self.threshold.items():
            threshold_length += 1


        print(threshold_length)

		
		
    def _initialize_weights2(self):

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,SeparatedBatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, dropout,batch_flag,bias,timesteps):

        if num_blocks==0:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout,batch_flag,bias,timesteps))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def network_update(self, timesteps):
        self.timesteps 	= timesteps
            
    def neuron_init(self, x):


        
        
        self.batch_size = x.size(0)
        self.width 		= x.size(2)
        self.height 	= x.size(3)

        self.mem 	= {}
        # self.spike 	= {}
        self.mask 	= {}


        #conv1
        pos = 0
        self.mem[0] = torch.zeros(self.batch_size,self.conv1.out_channels, self.width, self.height)
        # self.spike[0] = torch.ones(self.mem[pos].shape)*(-1000)

        pos += 1


        for i in range(1,4):
            layer = self.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l],nn.Conv2d):
                        self.width = self.width//layer[index].residual[l].stride[0]
                        self.height = self.height//layer[index].residual[l].stride[0]
                        self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height)
                        # self.spike[pos] = torch.ones(self.mem[pos].shape)*(-1000)
                        pos = pos + 1
                    elif isinstance(layer[index].residual[l],nn.Dropout):
                        self.mask[pos-1] = layer[index].residual[l](torch.ones(self.mem[pos-1].shape))
        

        #linear
        self.mem[pos] 	= torch.zeros(self.batch_size,self.labels)
        # self.spike[pos] 	= torch.ones(self.mem[pos].shape)*(-1000)



    def forward(self, x, find_max_mem=False, max_mem_layer=0):
        

        self.neuron_init(x)
            
        max_mem = 0.0
        #pdb.set_trace()

        for t in range(self.timesteps):
            # print(t)
            
            t_th = int(t//self.t_divide)
            pos=0

            out_prev = x

            #conv1

            mem_thr 			= (self.mem[pos]/getattr(self.threshold,'t'+str(t_th).zfill(2)+'_'+str(pos).zfill(2))) - 1.0
            out 				= self.act_func(mem_thr)
            rst 				= getattr(self.threshold, 't'+str(t_th).zfill(2)+'_'+str(pos).zfill(2)) * (mem_thr>0).float()
            # self.spike[pos] 	= self.spike[pos].masked_fill(out.bool(),t-1)
            self.mem[pos] 	    = getattr(self.leak, 'l'+str(t_th).zfill(2)+'_'+str(pos).zfill(2))*self.mem[pos] + self.conv1(out_prev) - rst
            out_prev  			= out.clone()  
 
            pos +=1
            
            for i in range(1,4):
                layer = self.layers[i]
                for index in range(len(layer)):
                    out_prev = layer[index]({'out_prev':out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem,  'mask':self.mask, 'threshold':self.threshold, 't': t, 't_th': t_th,'leak':self.leak ,'find_max_mem':find_max_mem})
                    pos = pos+2
  

            out = self.bn(out_prev,t)
            out = self.act_func(out)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0),-1)
            self.mem[pos] 			= self.mem[pos] + self.linear(out)



        return self.mem[pos]	


	























