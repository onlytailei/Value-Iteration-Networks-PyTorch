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
import torch.nn as nn
from utils import *

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

model = VIN_Block(args)
optimizer = optim.RMSprop(model.parameters(), args.lr)
criterion = nn.CrossEntropyLoss()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
model.type(dtype)

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(
        input=args.input_path, 
        imsize=args.imsize)
batch_size = args.batchsize

print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
for epoch in range(args.epochs):
    tstart = time.time()
    avg_err, avg_cost = 0.0, 0.0
    num_batches = int(Xtrain.shape[0]/batch_size)     
    for i in range(0, Xtrain.shape[0], batch_size):
        j = i+batch_size
        if j <= Xtrain.shape[0]:
            #print (Xtrain.dtype)
            X = torch.from_numpy(
                    np.transpose(Xtrain[i:j].astype(float),[0,3,1,2]))
            S1 = S1train[i:j]
            S2 = S2train[i:j]
            y_origin = ytrain[i * args.statebatchsize:j * 
                    args.statebatchsize].astype(np.int64)
            y = torch.from_numpy(y_origin)

            output,prediction = model(X, S1, S2)
            loss = criterion(output,autograd.Variable(y))    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cp = np.argmax(prediction.data.numpy(),1)
            err = np.mean(cp!=y_origin)
            avg_cost+=loss.data.numpy()[0]
            avg_err+=err
    
    if epoch % args.display_step == 0:
        elapsed = time.time() - tstart
        print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))

#test mode
Xtest_ = torch.from_numpy(np.transpose(Xtest.astype(float),[0,3,1,2]))
ytest_ = torch.from_numpy(ytest.astype(np.int64))
output_test,prediction_test = model(Xtest_, S1test, S2test)
cp_test = np.argmax(prediction_test.data.numpy(),1)
acc = np.mean(cp_test!=ytest)
print("Accuracy: {}%".format(100 * (1 - acc)))
