#!/usr/bin/env python
# coding=utf-8

'''
Author:Tai Lei
Date:Thu 09 Mar 2017 04:37:17 PM WAT
Info: Implement VIN through pytorch
'''

from __future__ import print_function
import time
import numpy as np
import torch
import argparse
from model import VIN_Block
from data import *
import torch.optim as optim
import torch.autograd as autograd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type = str, 
        default = 'data/gridworld_8.mat',
        help = "path of the training data")
parser.add_argument("--imsize", type = int, default = 8,
        help = "size of the input image")
parser.add_argument('--lr', type = float, default = 0.001,    
        help = 'Learning rate for RMSProp')
parser.add_argument('--epochs', type = int, default = 30,    
        help = 'Maximum epochs to train for')
parser.add_argument('--k', type = int, default = 10, 
        help = 'Number of value iterations')
parser.add_argument('--ch_i', type = int, default = 2,
        help = 'Channels in input layer')
parser.add_argument('--ch_h', type = int, default=150,
        help = 'Channels in initial hidden layer')
parser.add_argument('--ch_q', type = int, default = 10,
        help = 'Channels in q layer (~actions)')
parser.add_argument('--batchsize', type = int, default = 12, 
        help = 'Batch size')
parser.add_argument('--statebatchsize', type = int, default=10,
        help='Number of state inputs for each sample (real number, technically is k+1)')
parser.add_argument('--untied_weights', type = bool, default=False,  
        help = 'Untie weights of VI network')
parser.add_argument('--display_step', type = int,default=1,             
        help='Print summary output every n epochs')
parser.add_argument('--log', type = bool, default = False,          
        help = 'Enable for tensorboard summary')
parser.add_argument('--logdir', type = str, 
        default = '/tmp/vintf/',        
        help = 'Directory to store tensorboard summary')

args = parser.parse_args()

# Input tensor 
# Input batches of vertical positions (in different states)
# Input batches of horizental positions (in different states)

#build the model
model = VIN_Block(args)
#print (model.parameters())
optimizer = optim.RMSprop(model.parameters())

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=args.input_path, imsize=args.imsize)

batch_size = args.batchsize


for epoch in range(args.epochs):
    tstart = time.time()
    num_batches = int(Xtrain.shape[0]/batch_size)     
    for i in range(0, Xtrain.shape[0], batch_size):
        j = i+batch_size
        if j <= Xtrain.shape[0]:
            #print (Xtrain.dtype)
            X = torch.from_numpy(np.transpose(Xtrain[i:j].astype(float),[0,3,1,2]))
            S1 = S1train[i:j]
            S2 = S2train[i:j]
            y = torch.from_numpy(ytrain[i * args.statebatchsize:j * args.statebatchsize].astype(float))
            model(X, S1, S2)
            break
    break
