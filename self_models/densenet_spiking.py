#python3 snn.py -a DENSENET --optimizer Adam --dropout 0.3 --scaling_factor 0.3  --weight_decay 0.0  --lr_reduce 10  --store_name batch_type4_dropout03_epoch150 --lr_interval '0.3 0.60 0.70' --lr_reduce 500 --timesteps 40  --epochs 500 -lr 0.005 --default_threshold 1


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb
import math
from collections import OrderedDict
import copy
from .util import LinearSpike1,SeparatedBatchNorm1d


# __all__ = ["densenet100bc", "densenet190bc"]


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate,timesteps):
        super(Bottleneck, self).__init__()

        self.dense = nn.Sequential(
             SeparatedBatchNorm1d(num_features=in_planes, max_length=timesteps),
             nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False),
             SeparatedBatchNorm1d(num_features=4 * growth_rate, max_length=timesteps),
             nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )
        )


    def forward(self, dic):
        out_prev 		= dic['out_prev']
        pos 			= dic['pos']
        act_func 		= dic['act_func']
        mem 			= dic['mem']
        threshold 		= dic['threshold']
        t 				= dic['t']
        leak			= dic['leak']
        inp				= out_prev.clone()

        #bn
        batch_norm1 = self.dense[0](inp,t)

        #LinearSpike
        out1 			= act_func(batch_norm1)
        out_prev1  		= out1.clone()

        #conv+STDB
        conv1 = self.dense[1](out_prev1)
        mem_thr 		= (mem[pos]/getattr(threshold,'t'+str(t)+'_' +str(pos))) - 1.0
        rst 			= getattr(threshold,'t'+str(t)+'_' + str(pos)) * (mem_thr>0).float()
        # print(1)
        # print(pos)
        # print(mem[pos].shape)
        # print(mem_thr.shape)
        # print(rst.shape)
        # print(conv1.shape)
        mem[pos] 		= getattr(leak, 'l'+str(t)+'_'+str(pos)) *mem[pos] + conv1 - rst

        #bn
        batch_norm2 = self.dense[2](mem_thr,t)

        #LinearSpike
        out2 			= act_func(batch_norm2)
        out_prev2  		= out2.clone()

        #conv+STDB
        conv2 = self.dense[3](out_prev2)
        mem_thr 		= (mem[pos+1]/getattr(threshold, 't'+str(t)+'_'+str(pos+1))) - 1.0
        rst 			= getattr(threshold, 't'+str(t)+'_'+str(pos+1)) * (mem_thr>0).float()
        mem[pos+1] 		= getattr(leak,'l'+str(t)+'_'+str(pos+1))*mem[pos+1] + conv2 - rst

        # print(2)
        # print(pos+1)
        # print(mem[pos+1].shape)
        # print(mem_thr.shape)
        # print(rst.shape)
        # print(conv2.shape)

        out = torch.cat([mem_thr, inp], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes,timesteps):
        super(Transition, self).__init__()
        self.trans = nn.Sequential(
        SeparatedBatchNorm1d(num_features=in_planes, max_length=timesteps),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        )

    def forward(self,dic):
        out_prev 		= dic['out_prev']
        pos 			= dic['pos']
        act_func 		= dic['act_func']
        mem 			= dic['mem']
        threshold 		= dic['threshold']
        t 				= dic['t']
        leak			= dic['leak']
        inp				= out_prev.clone()

        #bn
        batch_norm = self.trans[0](inp,t)

        #LinearSpike
        out 			= act_func(batch_norm)
        out_prev  		= out.clone()

        #conv+STDB
        conv = self.trans[1](out_prev)
        mem_thr 		= (mem[pos]/getattr(threshold,'t'+str(t)+'_' +str(pos))) - 1.0
        rst 			= getattr(threshold,'t'+str(t)+'_' + str(pos)) * (mem_thr>0).float()
        mem[pos] 		= getattr(leak, 'l'+str(t)+'_'+str(pos)) *mem[pos] + conv - rst

        #avg pooling
        # print(mem_thr.shape)
        out = F.avg_pool2d(mem_thr, 2)
        # print(out.shape)
        return out


class DENSENET_SNN(nn.Module):
    def __init__(self, depth, growth_rate=12, reduction=0.5, num_classes=10,timesteps=40,leak=1,default_threshold=1):
        super().__init__()

        self.act_func 		         = LinearSpike1.apply
        self.timesteps 		         = timesteps
        self.mem 			         = {}
        self.mask 			         = {}
        self.pre_process_batch_norm  = {}

        self.timesteps = timesteps
        self.growth_rate = growth_rate
        nblocks = (depth - 4) // 6
        num_planes = 2 * growth_rate
        self.conv_1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        block = Bottleneck

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks)
        # print(num_planes)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_1 = Transition(num_planes, out_planes,self.timesteps)
        num_planes = out_planes
        # print(num_planes)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_2 = Transition(num_planes, out_planes,self.timesteps)
        num_planes = out_planes
        # print(num_planes)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        # print(num_planes)

        self.bn = SeparatedBatchNorm1d(num_features=num_planes, max_length=timesteps)
        self.fc = nn.Linear(num_planes, num_classes)


        self.dense_layers = {0: self.dense1, 1: self.dense2, 2: self.dense3}
        self.trans_layers = {0: self.trans_1, 1: self.trans_2}
        # print(self.dense_layers[1][0])
        # print(self.dense_layers[1][1])
        # print(self.dense_layers[1][2])


        # print(self.dense_layers[0][0].dense[1].out_channels)
        # print(self.dense_layers[0][1].dense[1].out_channels)
        # print(self.dense_layers[0][2].dense[1].out_channels)
        # exit(1)


        self._initialize_weights2()

        threshold 	= {}
        lk 			= {}

        for t in range(self.timesteps):


            pos = 0
            #conv_1
            threshold['t'+str(t)+'_'+str(pos)] 	= nn.Parameter(torch.Tensor(default_threshold))
            lk['l'+str(t)+'_'+str(pos)] 	  		= nn.Parameter(torch.Tensor(leak))
            pos += 1

            for i in range(0,3):


                layer_dense = self.dense_layers[i]
                layer_trans = self.trans_layers[i] if i != 2 else None
                for index in range(len(layer_dense)):

                    #dense
                    # make_layer has multi dense_layer
                    for l in range(len(layer_dense[index].dense)):
                        if isinstance(layer_dense[index].dense[l],nn.Conv2d):

                            threshold['t'+str(t)+'_'+str(pos)] = nn.Parameter(torch.Tensor(default_threshold))
                            lk['l'+str(t)+'_'+str(pos)] 		= nn.Parameter(torch.Tensor(leak))
                            pos += 1

                #trans
                if(i != 2):
                    for l1 in range(len(layer_trans.trans)):
                        if isinstance(layer_trans.trans[l1],nn.Conv2d):
                            threshold['t'+str(t)+'_'+str(pos)] = nn.Parameter(torch.Tensor(default_threshold))
                            lk['l'+str(t)+'_'+str(pos)] 		= nn.Parameter(torch.Tensor(leak))
                            pos += 1
                
                
            
            # print(pos)




            self.threshold 	= nn.ParameterDict(threshold)
            self.leak 		= nn.ParameterDict(lk)


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate,self.timesteps))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)


    def _initialize_weights2(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,SeparatedBatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def network_update(self, timesteps):
        self.timesteps 	= timesteps

    def neuron_init(self, x):

        self.batch_size = x.size(0)
        self.width 		= x.size(2)
        self.height 	= x.size(3)

        self.mem 	= {}
        self.mask 	= {}

        pos = 0


        #conv_1

        self.mem[pos] = torch.zeros(self.batch_size,self.conv_1.out_channels, self.width, self.height)

        pos += 1

        rate = 1
        for i in range(0,3):
            
            rate = 2*i if i != 0 else 1
            layer_dense = self.dense_layers[i]
            layer_trans = self.trans_layers[i] if i != 2 else None
            for index in range(len(layer_dense)):
                
                #dense
                # print(self.width/rate)
                for l in range(len(layer_dense[index].dense)):
                    if isinstance(layer_dense[index].dense[l],nn.Conv2d):
                        self.mem[pos] = torch.zeros(self.batch_size,layer_dense[index].dense[l].out_channels, int(self.width/rate), int(self.height/rate))
                        # print(index)
                        # print(layer_dense[index].dense[l].out_channels)
                        # print(l,index)
                        # print("ini {}".format(self.mem[pos].shape))
                        # print(pos)
                        pos += 1
                        # print(pos)

                # exit(1)

            #trans
            if(i != 2):
                for l1 in range(len(layer_trans.trans)):
                    if isinstance(layer_trans.trans[l1],nn.Conv2d):
                        self.mem[pos] = torch.zeros(self.batch_size,layer_trans.trans[l1].out_channels, int(self.width/rate), int(self.height/rate))
                        pos += 1

                # print(pos)
                # for i in range(50):
                #     print(self.mem[i].shape)



        self.mem[pos] 	= torch.zeros(self.batch_size,self.fc.out_features)








    def forward(self, x):
        # out = self.conv_1(x)
        # out = self.trans_1(self.dense1(out))
        # out = self.trans_2(self.dense2(out))
        # out = self.dense3(out)
        # out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        # out = out.view(out.size(0), -1)
        # out = self.fc ...

        self.neuron_init(x)
        #pdb.set_trace()
        for t in range(self.timesteps):
            #conv+STDB
            pos = 0
            out = self.conv_1(x)
            mem_thr 		= (self.mem[pos]/getattr(self.threshold,'t'+str(t)+'_' +str(pos))) - 1.0
            rst 			= getattr(self.threshold,'t'+str(t)+'_' + str(pos)) * (mem_thr>0).float()
            self.mem[pos] 		= getattr(self.leak, 'l'+str(t)+'_'+str(pos)) *self.mem[pos] + out - rst
            pos += 1
            #dense1
            # print(mem_thr.shape)[]
            out_prev = mem_thr
            for i in range(0,3):

                #dense
                layer1 = self.dense_layers[i]
                for index in range(len(layer1)):
                    # print(index)
                    # print(pos)
                    # print(out_prev.shape)
                    out_prev = layer1[index]({'out_prev':out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem, 'threshold':self.threshold, 't': t, 'leak':self.leak})
                    pos = pos+2
                #trans
                layer2 = self.trans_layers[i] if i != 2 else None
                if (i!=2):
                    out_prev = layer2({'out_prev':out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem, 'threshold':self.threshold, 't': t, 'leak':self.leak})
                    pos = pos+1
                    # print(out_prev.shape)
                


            # #trans1
            # out = self.trans_1({'out_prev':out.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem,'threshold':self.threshold, 't': t, 'leak':self.leak })
            # pos += 1

            # #dense2
            # out = self.dense1({'out_prev':out.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem,'threshold':self.threshold, 't': t, 'leak':self.leak })
            # pos += 2
            # #trans2
            # out = self.trans_1({'out_prev':out.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem,'threshold':self.threshold, 't': t, 'leak':self.leak })
            # pos += 1

            # #dense3
            # out = self.dense1({'out_prev':out.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem,'threshold':self.threshold, 't': t, 'leak':self.leak })
            # pos += 2


            out = self.bn(out_prev,t)

            out 			= self.act_func(out)
            out_prev  		= out.clone()



            out = F.avg_pool2d(out_prev, 8)

            out = out.view(out.size(0), -1)


            self.mem[pos] = self.mem[pos] + self.fc(out)

        return self.mem[pos]




# def densenet100bc(num_classes):
#     return DenseNet(Bottleneck, depth=100, growth_rate=12, num_classes=num_classes)


# def densenet190bc(num_classes):
#     return DenseNet(Bottleneck, depth=190, growth_rate=40, num_classes=num_classes)