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
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride, dropout,batch_flag,bias,timesteps):
		#print('In __init__ BasicBlock')
		#super(BasicBlock, self).__init__()
		super().__init__()
		self.identity_conv_flag = False
		if batch_flag:
			self.residual = nn.Sequential(
				nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
				nn.BatchNorm2d(planes),
				nn.ReLU(inplace=True),
				nn.Dropout(dropout),
				nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(planes),
				)
			self.identity = nn.Sequential()
			if stride != 1 or in_planes != self.expansion*planes:
				self.identity = nn.Sequential(
					nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(self.expansion*planes)
				)
		else:
			self.residual = nn.Sequential(
				nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias),
				#nn.BatchNorm2d(planes),
				nn.ReLU(inplace=True),
				nn.Dropout(dropout),
				nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias),
				#nn.BatchNorm2d(planes),
				)
			self.identity = nn.Sequential()
			if stride != 1 or in_planes != self.expansion*planes:
				self.identity_conv_flag = True
				self.identity = nn.Sequential(
					nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
					#nn.BatchNorm2d(self.expansion*planes)
				)
		
		self.residual_batch0 = SeparatedBatchNorm1d(num_features=planes, max_length=timesteps)
		self.residual_batch3 = SeparatedBatchNorm1d(num_features=planes, max_length=timesteps)
		self.identity_batch= SeparatedBatchNorm1d(num_features=self.expansion*planes, max_length=timesteps)






	def forward(self, dic):
		
		out_prev 		= dic['out_prev']
		pos 			= dic['pos']
		act_func 		= dic['act_func']
		mem 			= dic['mem']
		spike 			= dic['spike']
		mask 			= dic['mask']
		threshold 		= dic['threshold']
		t 				= dic['t']
		leak			= dic['leak']
		#find_max_mem 	= dic['find_max_mem']
		inp				= out_prev.clone()
		
		#conv1
		mem_thr 		= (mem[pos]/getattr(threshold, 't'+str(pos))) - 1.0
		rst 			= getattr(threshold, 't'+str(pos)) * (mem_thr>0).float()
		residual_batch  = self.residual_batch0(self.residual[0](inp),t)
		mem[pos] 		= getattr(leak, 'l'+str(pos)) *mem[pos] + residual_batch - rst

		#batch normalization


		#relu1
		out 			= act_func(mem_thr, (t-1-spike[pos]))
		spike[pos] 		= spike[pos].masked_fill(out.bool(),t-1)
		out_prev  		= out.clone()

		#dropout1
		out_prev 		= out_prev * mask[pos]
		
		#conv2+identity
		mem_thr 		= (mem[pos+1]/getattr(threshold, 't'+str(pos+1))) - 1.0
		rst 			= getattr(threshold, 't'+str(pos+1)) * (mem_thr>0).float()
		residual_batch  = self.residual_batch3(self.residual[3](out_prev),t)
		identity  = self.identity_batch(self.identity(inp),t) if self.identity_conv_flag else self.identity(inp)
		mem[pos+1] 		= getattr(leak, 'l'+str(pos+1))*mem[pos+1] + residual_batch + identity - rst

		#relu2
		out 			= act_func(mem_thr, (t-1-spike[pos+1]))
		spike[pos+1]	= spike[pos+1].masked_fill(out.bool(),t-1)
		out_prev  		= out.clone()

		return out_prev

class RESNET_SNN_BATCH_NORM(nn.Module):
	
	#all_layers = []
	#drop 		= 0.2
	def __init__(self, resnet_name, activation='Linear', labels=10, timesteps=75, leak=1.0, default_threshold=1.0, dropout=0.2, dataset='CIFAR10',batch_flag=False,bias=False):

		super().__init__()
		
		self.resnet_name	         = resnet_name.lower()
		self.act_func 		         = LinearSpike.apply
		self.labels 		         = labels
		self.timesteps 		         = timesteps
		self.dropout 		         = dropout
		self.dataset 		         = dataset
		self.mem 			         = {}
		self.mask 			         = {}
		self.spike 			         = {}
		self.pre_process_batch_norm  = {}


		
		if batch_flag:
			self.pre_process = nn.Sequential(
									nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
									nn.BatchNorm2d(64),
									nn.ReLU(inplace=True),
									nn.Dropout(self.dropout),
									nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
									nn.BatchNorm2d(64),
									nn.ReLU(inplace=True),
									nn.Dropout(self.dropout),
									nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
									nn.BatchNorm2d(64),
									nn.ReLU(inplace=True),
									nn.AvgPool2d(2)
									)
		else:
			self.pre_process = nn.Sequential(
									nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias),
									nn.ReLU(inplace=True),
									nn.Dropout(self.dropout),
									nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),
									nn.ReLU(inplace=True),
									nn.Dropout(self.dropout),
									nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),
									nn.ReLU(inplace=True),
									nn.AvgPool2d(2)
									)
					


		block               = BasicBlock
		self.in_planes      = 64


		self.layer1 		= self._make_layer(block, 64,  cfg[self.resnet_name][0], stride=1, dropout=self.dropout\
			,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
		self.layer2 		= self._make_layer(block, 128, cfg[self.resnet_name][1], stride=2, dropout=self.dropout\
			,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
		self.layer3 		= self._make_layer(block, 256, cfg[self.resnet_name][2], stride=2, dropout=self.dropout\
			,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
		self.layer4 		= self._make_layer(block, 512, cfg[self.resnet_name][3], stride=2, dropout=self.dropout\
			,batch_flag=batch_flag,bias=bias,timesteps=self.timesteps)
		#self.avgpool 		= nn.AvgPool2d(2)
		
		self.classifier     = nn.Sequential(
									nn.Linear(512*2*2, labels, bias=False)
									)
		
		self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4:self.layer4}

		self.pre_process_batch0 = SeparatedBatchNorm1d(num_features=64, max_length=self.timesteps)
		self.pre_process_batch3 = SeparatedBatchNorm1d(num_features=64, max_length=self.timesteps)
		self.pre_process_batch6 = SeparatedBatchNorm1d(num_features=64, max_length=self.timesteps)

		self.pre_process_batch_norm = {0:self.pre_process_batch0,3:self.pre_process_batch3,6:self.pre_process_batch6}

		self._initialize_weights2()
		
		threshold 	= {}
		lk 			= {}
		for l in range(len(self.pre_process)):
			if isinstance(self.pre_process[l],nn.Conv2d):
				threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(l)] 	  		= nn.Parameter(torch.tensor(leak))

		pos = len(self.pre_process)
				
		for i in range(1,5):

			layer = self.layers[i]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						threshold['t'+str(pos)] = nn.Parameter(torch.tensor(default_threshold))
						lk['l'+str(pos)] 		= nn.Parameter(torch.tensor(leak))
						pos=pos+1

		for l in range(len(self.classifier)-1):
			if isinstance(self.classifier[l], nn.Linear):
				threshold['t'+str(pos+l)] 		= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(pos+l)] 				= nn.Parameter(torch.tensor(leak)) 
				
		self.threshold 	= nn.ParameterDict(threshold)
		self.leak 		= nn.ParameterDict(lk)

		
		
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

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):
    	
		self.scaling_factor = scaling_factor
			
		for pos in range(len(self.pre_process)):
			if isinstance(self.pre_process[pos],nn.Conv2d):
				if thresholds:
					self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scaling_factor))})

		pos = len(self.pre_process)
		for i in range(1,5):
			layer = self.layers[i]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						
						pos = pos+1

		for l in range(len(self.classifier)):
			if isinstance(self.classifier[l], nn.Linear):
				if thresholds:
					self.threshold.update({'t'+str(pos+l): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scaling_factor))})
			

	def _make_layer(self, block, planes, num_blocks, stride, dropout,batch_flag,bias,timesteps):

		if num_blocks==0:
			return nn.Sequential()
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, dropout,batch_flag,bias,timesteps))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
			
	def neuron_init(self, x):
		
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)

		self.mem 	= {}
		self.spike 	= {}
		self.mask 	= {}

		# Pre process layers
		for l in range(len(self.pre_process)):
			
			if isinstance(self.pre_process[l], nn.Conv2d):
				self.mem[l] = torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height)
				self.spike[l] = torch.ones(self.mem[l].shape)*(-1000)
				
			elif isinstance(self.pre_process[l], nn.Dropout):
				self.mask[l] = self.pre_process[l](torch.ones(self.mem[l-2].shape))
			elif isinstance(self.pre_process[l], nn.AvgPool2d):
				
				self.width 	= self.width//self.pre_process[l].kernel_size
				self.height = self.height//self.pre_process[l].kernel_size 

		pos = len(self.pre_process)
		for i in range(1,5):
			layer = self.layers[i]
			self.width = self.width//layer[0].residual[0].stride[0]
			self.height = self.height//layer[0].residual[0].stride[0]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height)
						self.spike[pos] = torch.ones(self.mem[pos].shape)*(-1000)
						pos = pos + 1
					elif isinstance(layer[index].residual[l],nn.Dropout):
						self.mask[pos-1] = layer[index].residual[l](torch.ones(self.mem[pos-1].shape))
		
		for l in range(len(self.classifier)):
			if isinstance(self.classifier[l],nn.Linear):
				self.mem[pos+l] 	= torch.zeros(self.batch_size, self.classifier[l].out_features)
				self.spike[pos+l] 	= torch.ones(self.mem[pos+l].shape)*(-1000)
			elif isinstance(self.classifier[l], nn.Dropout):
				self.mask[pos+l] 	= self.classifier[l](torch.ones(self.mem[pos+l-2].shape))

	def percentile(self, t, q):
		k = 1 + round(.01 * float(q) * (t.numel() - 1))
		result = t.view(-1).kthvalue(k).values.item()
		return result

	def forward(self, x, find_max_mem=False, max_mem_layer=0):
		
		self.neuron_init(x)
			
		max_mem = 0.0
		#pdb.set_trace()
		for t in range(self.timesteps):

			out_prev = x
					
			for l in range(len(self.pre_process)):
							
				if isinstance(self.pre_process[l], nn.Conv2d):

					
					if find_max_mem and l==max_mem_layer:
						cur = self.percentile(self.pre_process[l](out_prev).view(-1), 99.7)
						if (cur>max_mem):
							max_mem = torch.tensor([cur])
						break

					mem_thr 		= (self.mem[l]/getattr(self.threshold, 't'+str(l))) - 1.0
					rst 			= getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					pre_process_conv_batch = self.pre_process_batch_norm[l](self.pre_process[l](out_prev),t)
					self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + pre_process_conv_batch- rst
					
				elif isinstance(self.pre_process[l], nn.ReLU):
					out 			= self.act_func(mem_thr, (t-1-self.spike[l-1]))
					self.spike[l-1] = self.spike[l-1].masked_fill(out.bool(),t-1)
					out_prev  		= out.clone()

				elif isinstance(self.pre_process[l], nn.AvgPool2d):
					out_prev 		= self.pre_process[l](out_prev)
				
				elif isinstance(self.pre_process[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l]

				elif isinstance(self.pre_process[l], nn.Dropout):
					out_prev 		= self.pre_process[l](out_prev)

				
			
			if find_max_mem and max_mem_layer<len(self.pre_process):
				continue
				
			pos 	= len(self.pre_process)
			
			for i in range(1,5):
				layer = self.layers[i]
				for index in range(len(layer)):
					out_prev = layer[index]({'out_prev':out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem, 'spike':self.spike, 'mask':self.mask, 'threshold':self.threshold, 't': t, 'leak':self.leak})
					pos = pos+2
			
			#out_prev = self.avgpool(out_prev)
			out_prev = out_prev.reshape(self.batch_size, -1)

			for l in range(len(self.classifier)-1):
				
				if isinstance(self.classifier[l], (nn.Linear)):
					if find_max_mem and (pos+l)==max_mem_layer:
						if (self.classifier[l](out_prev)).max()>max_mem:
							max_mem = (self.classifier[l](out_prev)).max()
						break

					mem_thr 			= (self.mem[pos+l]/getattr(self.threshold, 't'+str(pos+l))) - 1.0
					out 				= self.act_func(mem_thr, (t-1-self.spike[pos+l]))
					rst 				= getattr(self.threshold, 't'+str(pos+l)) * (mem_thr>0).float()
					self.spike[pos+l] 	= self.spike[pos+l].masked_fill(out.bool(),t-1)
					self.mem[pos+l] 	= getattr(self.leak, 'l'+str(pos+l))*self.mem[pos+l] + self.classifier[l](out_prev) - rst
					out_prev  			= out.clone()

				elif isinstance(self.classifier[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[pos+l]

			#pdb.set_trace()
			# Compute the final layer outputs
			if not find_max_mem:
				if len(self.classifier)>1:
					self.mem[pos+l+1] 		= self.mem[pos+l+1] + self.classifier[l+1](out_prev)
				else:
					self.mem[pos] 			= self.mem[pos] + self.classifier[0](out_prev)
		
		if find_max_mem:
			return max_mem

		if len(self.classifier)>1:
			return self.mem[pos+l+1]
		else:
			return self.mem[pos]	


	























