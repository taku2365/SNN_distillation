import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

	
if __name__ == "__main__":
	pass