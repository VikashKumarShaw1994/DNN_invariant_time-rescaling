# %%
#%matplotlib inline

# %%
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_context("poster")

import torch
from torch import nn as nn
ttype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
ctype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
print(ttype)
import torch.nn.functional as F
from matplotlib import gridspec
from SITHCon.sithcon import SITHCon_Layer, _SITHCon_Core, iSITH

from tqdm import tqdm

import itertools
from csv import DictWriter
import os 
from os.path import join
import glob

import numpy as np
import pandas as pd
import pickle
from math import factorial
import random

# %%
MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 
                    'C':'-.-.', 'D':'-..', 'E':'.', 
                    'F':'..-.', 'G':'--.', 'H':'....', 
                    'I':'..', 'J':'.---', 'K':'-.-', 
                    'L':'.-..', 'M':'--', 'N':'-.', 
                    'O':'---', 'P':'.--.', 'Q':'--.-', 
                    'R':'.-.', 'S':'...', 'T':'-', 
                    'U':'..-', 'V':'...-', 'W':'.--', 
                    'X':'-..-', 'Y':'-.--', 'Z':'--..', 
                    '1':'.----', '2':'..---', '3':'...--', 
                    '4':'....-', '5':'.....', '6':'-....', 
                    '7':'--...', '8':'---..', '9':'----.', 
                    '0':'-----', ', ':'--..--', '.':'.-.-.-', 
                    '?':'..--..', '/':'-..-.', '-':'-....-', 
                    '(':'-.--.', ')':'-.--.-'} 

# %%
morse_code_numpy = {key:np.array([int(x) for x in MORSE_CODE_DICT[key].replace('.', '10').replace('-', '1110')] + [0, 0])
                    for key in MORSE_CODE_DICT.keys()}

for k in morse_code_numpy.keys():
    print(morse_code_numpy[k], k)
subset = list(morse_code_numpy.keys())

id2key = subset
key2id = {}
for idx, s in enumerate(subset):
    key2id[s] = idx

X = [ttype(morse_code_numpy[k])for k in subset]
Y = torch.LongTensor(np.arange(0,len(X)))
print(X, Y)

# %%
class SITHCon_Classifier(nn.Module):
    def __init__(self, out_classes, layer_params, 
                 act_func=nn.ReLU):
        super(SITHCon_Classifier, self).__init__()
        last_channels = layer_params[-1]['channels']
        self.transform_linears = nn.ModuleList([nn.Linear(l['channels'], l['channels'])
                                                for l in layer_params])
        self.sithcon_layers = nn.ModuleList([SITHCon_Layer(l, act_func) for l in layer_params])
        self.to_out = nn.Linear(last_channels, out_classes)
        
        
    def forward(self, inp):
        
        x = inp
        #out = []
        for i in range(len(self.sithcon_layers)):
            x = self.sithcon_layers[i](x)
            
            x = self.transform_linears[i](x[:,0,:,:].transpose(1,2))
            x = x.unsqueeze(1).transpose(2,3)

            #out.append(x.clone())
        x = x.transpose(2,3)[:, 0, :, :]
        #x = x.transpose(2,3)[:, 0, :, :]
        x = self.to_out(x)
        return x

# %% [markdown]
# # Three Layers

# %%
p = [400, 35, 23, 2]

sp1 = dict(in_features=1, 
           tau_min=.1, tau_max=4000, buff_max=6500,
           dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype, 
           channels=35, kernel_width=p[2], dilation=p[3],
           dropout=None, batch_norm=None)
'''sp1 = dict(in_features=1,
           tau_min=.1, tau_max=4000, buff_max=6500,
           dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype,
           channels=35, kernel_width=p[2], dilation=p[3])'''
sp2 = dict(in_features=sp1['channels'], 
           tau_min=.1, tau_max=4000, buff_max=6500,
           dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype, 
           channels=35, kernel_width=p[2], dilation=p[3], 
           dropout=None, batch_norm=None)
'''sp2 = dict(in_features=sp1['channels'],
           tau_min=.1, tau_max=4000, buff_max=6500,
           dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype,
           channels=35, kernel_width=p[2], dilation=p[3])'''
sp3 = dict(in_features=sp2['channels'], 
           tau_min=.1, tau_max=4000, buff_max=6500,
           dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype, 
           channels=35, kernel_width=p[2], dilation=p[3], 
           dropout=None, batch_norm=None)

# TWO LAYERS
layer_params = [sp1, sp2]#, sp3]


model = SITHCon_Classifier(len(X), layer_params, act_func=nn.ReLU)
model
tot_weights = 0
for p in model.parameters():
    tot_weights += p.numel()
print("Total Weights:", tot_weights)
print(model)

# %%
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 5
Trainscale = 10
device='cpu'
batch_size = 8
batches = int(np.ceil(43 / batch_size))
progress_bar = tqdm(range(int(epochs)), bar_format='{l_bar}{bar:5}{r_bar}{bar:-5b}')
times_100 = 0
for epoch_idx in progress_bar:
    perfs = []
    losses = []
    model.train()
    for batch_idx in range(batches):
        optimizer.zero_grad()
        loss = 0
        permute = np.arange(0, 43)
        for i in range(0, int(min(len(X) - (batch_idx*batch_size), 
                              batch_size))
                       ):
            iv = X[permute[batch_idx*batch_size + i]]
            iv = iv.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            iv = iv.unsqueeze(-1)
            iv = iv.repeat(1,1,1,1,Trainscale)
            iv = iv.reshape(1,1,1,-1)
            tv = Y[permute[batch_idx*batch_size + i]].to(device)
            optimizer.zero_grad()
            out = model(iv)
            loss += loss_func(out[:, -1, :],
                             torch.LongTensor([tv]))
            perfs.append((torch.argmax(out[:, -1, :], dim=-1) == 
                      tv).sum().item())
        loss = loss / min(len(X) - (batch_idx*batch_size), 
                          batch_size)
        loss.backward()
        optimizer.step()
        
        
        #perfs = perfs[int(-loss_buffer_size/batch_size):]
        losses.append(loss.detach().cpu().numpy())
        #losses = losses[int(-loss_buffer_size/batch_size):]
        
        
        s = "{}:{:2} Loss: {:.4f}, Perf: {:.4f}"
        format_list = [epoch_idx, batch_idx, np.mean(losses), 
                       np.sum(perfs)/((len(perfs)))]
        s = s.format(*format_list)
        progress_bar.set_description(s)
    if (np.sum(perfs)/((len(perfs))) == 1.0) & (np.mean(losses) < .11):
        times_100 += 1
        if times_100 >= 3:
            break
    else:
        times_100 = 0

# %% [markdown]
# # TEST
# 

# %%
model.eval()
evald = []
evaldDict = {'perf':[],
             'scale':[]}
for nr in [1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40]:
#for nr in range(1,40,):
    perfs = []
    for batch_idx, iv in enumerate(X):
        iv = iv.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        iv = iv.unsqueeze(-1)
        iv = iv.repeat(1,1,1,1,nr)
        iv = iv.reshape(1,1,1,-1)
        tv = Y[batch_idx].to(device)
        out = model(iv)
        loss = loss_func(out[:, -1, :],
                         torch.LongTensor([tv]))


        perfs.append((torch.argmax(out[:, -1, :], dim=-1) == 
                      tv).sum().item())
    evaldDict['perf'].append(sum(perfs)/len(perfs))
    evaldDict['scale'].append(nr)
    print(nr, sum(perfs)/len(perfs))
    evald.append([nr, sum(perfs)/(len(perfs)*1.0)])
scale_perfs = pd.DataFrame(evaldDict)
directory = "perf"
if not os.path.exists(directory):
    os.makedirs(directory)
scale_perfs.to_pickle(join("perf", "sithcon_morse_test.dill"))


