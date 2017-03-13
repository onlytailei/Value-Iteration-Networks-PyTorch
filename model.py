#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu 09 Mar 2017 05:38:33 PM WAT
Info: VIN module 
'''
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import torch.nn.functional as F
# use torch functional layer here

def agLT(x):
    return autograd.Variable(torch.LongTensor([x]))

class VIN_Block(nn.Module):
    def __init__(self, arg):
        super(VIN_Block, self).__init__()
        self.k = arg.k
        self.ch_i = arg.ch_i
        self.ch_h = arg.ch_h
        self.ch_q = arg.ch_q
        self.state_batch_size = arg.statebatchsize 

        self.bias = Parameter(torch.zeros(self.ch_h).random_(0,1)*0.01)
        self.register_parameter('bias', self.bias)
        self.w0 = Parameter(torch.zeros(self.ch_h,self.ch_i,3,3).random_(0,1)*0.01)
        self.register_parameter('w0', self.w0)
        self.w1 = Parameter(torch.zeros(1,self.ch_h,1,1).random_(0,1)*0.01)
        self.register_parameter('w1', self.w1)
        self.w = Parameter(torch.zeros(self.ch_q,1,3,3).random_(0,1)*0.01)
        self.register_parameter('w', self.w)
        self.w_fb = Parameter(torch.zeros(self.ch_q,1,3,3).random_(0,1)*0.01)
        self.register_parameter('w_fb', self.w_fb)
        self.w_o = Parameter(torch.zeros(8, self.ch_q).random_(0,1)*0.01)
        self.register_parameter('w_o', self.w_o)
        self.softmax = nn.Softmax()

    def forward(self, X, S1, S2):
        X=autograd.Variable(X)
        h = F.conv2d(X.float(), self.w0, bias = self.bias, padding = 1)
        r = F.conv2d(h, self.w1)
        q = F.conv2d(r, self.w, padding = 1)
        v,_ = torch.max(q, 1) 

        for i in range(0, self.k-1):

            rv = torch.cat((r,v),1)
            wwfb = torch.cat((self.w, self.w_fb),1)

            q = F.conv2d(rv, wwfb, padding = 1)
            v,_ = torch.max(q,1) 

        q = F.conv2d(torch.cat((r,v),1),torch.cat((self.w,self.w_fb),1), padding = 1)

        bs = q.data.numpy().shape[0]
        len_ = self.state_batch_size*bs
        rprn = np.array([[item]*self.state_batch_size for item in np.arange(bs)],dtype=np.int64).reshape(len_)
        ins1 = S1.reshape(len_).astype(np.int64)
        ins2 = S2.reshape(len_).astype(np.int64)

        q_ = torch.transpose(q, 0,2)
        q__ = torch.transpose(q_,1 ,3)
        
        # TODO need to be optimize cause there is no gather_nd in pytorch
        abs_q = torch.index_select(
                    torch.index_select(
                        torch.index_select(q__,0,agLT(ins1[0])),
                    1,agLT(ins2[0])),
                2,agLT(rprn[0]))
        for item in np.arange(1,len_):
            abs_q_ = torch.index_select(
                    torch.index_select(
                        torch.index_select(q__,0,
                                agLT(ins1[item])),
                        1,agLT(ins2[item])),
                    2,agLT(rprn[item]))
            abs_q = torch.cat((abs_q,abs_q_),0)

        final_q = torch.squeeze(abs_q)
        output = F.linear(final_q, self.w_o)
        prediction = self.softmax(output)
        return output,prediction

if __name__ == "__main__":
    obj = VIN_Block()
    print (obj.parameters)
