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

from tqdm.notebook import tqdm

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
full_path = join('/media', 'root', 'easystore', 'experiments', 
                 'audiomnist', '1.0', 'AudioMNIST', 'data')

# %%
all_files = pd.read_csv(join('data','files_info.csv'))

tr_info = all_files[(all_files.rec_split<9)&(all_files.subjid!=60)].reset_index()
del tr_info['index']
tr_info['train_scale'] = [1.0]*tr_info.shape[0]

print(tr_info.head())

# %%
start_path ="/media/root/easystore/experiments/audiomnist/{:05.2f}/AudioMNIST/data/data_{:02}_{:.2f}.npy"
print("/media/root/easystore/experiments/audiomnist/{:05.2f}/AudioMNIST/data/data_{:02}_{:.2f}.npy".format(1.0, 
                                                                                                          1, 
                                                                                                          1.0))

# %%
scales = [.1,.4,  1.0,  2.5,10.]
nps = int(tr_info[(tr_info.digit==0)].count()[0]/len(scales))
num_per_scale = [nps]*len(scales)
print(tr_info[(tr_info.digit==0)].count()[0])
num_per_scale[-1] = num_per_scale[-1] + tr_info[(tr_info.digit==0)].count()[0] - nps*len(scales)
print(num_per_scale)


# %% [markdown]
# # DO NOT RUN. THIS WILL BE PROVIDED
# 
# 
#     # From digit 0 to 9
#     for d in range(0, 10):
#         idxs = tr_info[tr_info.digit == d].index
#         # randomly shuffle order so we get random recordings from random 
#         # participants getting scaled.
#         permute = np.arange(idxs.shape[0])
#         num_curr_scale = 0
#         scale_index = 0
#         for i in idxs[permute]:
#             tr_info.loc[i, 'train_scale'] = scales[scale_index]
#             num_curr_scale += 1
#             if num_curr_scale == num_per_scale[scale_index]:
#                 scale_index += 1
#                 num_curr_scale = 0
#         trainX = []
#         trainY = []
#     for scale in tr_info.train_scale.unique():
#         for subjid in tr_info.subjid.unique():
#             fp = start_path.format(scale, 
#                                    subjid, 
#                                    scale)
#             loaded = np.load(fp)
# 
#             dat_idxs = list(tr_info.loc[(tr_info.train_scale==scale)&
#                                     (tr_info.subjid==subjid), 
#                                     'dat_idx'])
# 
#             small_datX = [loaded[idx] for idx in dat_idxs]
#             small_datY = list(tr_info.loc[(tr_info.train_scale==scale)&
#                                           (tr_info.subjid==subjid), 
#                                           'digit'])
#             trainX += small_datX
#             trainY += small_datY
#             print('Finished {} at {}'.format(subjid, scale))
#     with open(join('data', 'trainX_sawtooth.dill'), "wb") as f:
#         pickle.dump(trainX, f)
#     with open(join('data', 'trainY_sawtooth.dill'), "wb") as f:
#         pickle.dump(trainY, f)
# 

# %%
tr_info.loc[(tr_info.train_scale==scale)&
                                (tr_info.subjid==subjid), 
                                ]

# %%
with open(join('data', 'trainX_sawtooth.dill'), "rb") as f:
    trainX = pickle.load(f)
with open(join('data', 'trainY_sawtooth.dill'), "rb") as f:
    trainY = pickle.load(f)


# %% [markdown]
# # Classes 

# %%
class SITHCon_Classifier(nn.Module):
    def __init__(self, out_classes, layer_params, 
                 act_func=nn.ReLU, batch_norm=False,
                 dropout=.2):
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
            
            x = F.relu(self.transform_linears[i](x[:,0,:,:].transpose(1,2)))
            x = x.unsqueeze(1).transpose(2,3)

            #out.append(x.clone())
        x = x.transpose(2,3)[:, 0, :, :]
        #x = x.transpose(2,3)[:, 0, :, :]
        x = self.to_out(x)
        return x

# %% [markdown]
# # Functions

# %%
def gen_model(p):
    sp1 = dict(in_features=50, 
               tau_min=1, tau_max=4000, buff_max=6500,
               dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype, 
               channels=35, kernel_width=p[2], dilation=p[3],
               dropout=None, batch_norm=None)
    sp2 = dict(in_features=sp1['channels'], 
               tau_min=1, tau_max=4000, buff_max=6500,
               dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype, 
               channels=35, kernel_width=p[2], dilation=p[3], 
               dropout=None, batch_norm=None)
    sp3 = dict(in_features=sp2['channels'], 
               tau_min=1, tau_max=4000, buff_max=6500,
               dt=1, ntau=p[0], k=p[1], g=0.0, ttype=ttype, 
               channels=35, kernel_width=p[2], dilation=p[3], 
               dropout=None, batch_norm=None)
    layer_params = [sp1, sp2]#, sp3]
    model = SITHCon_Classifier(10, layer_params, act_func=None,#nn.ReLU
                              ).cuda()
    return model

def train(model, ttype, trainX, trainY, testX, testY, optimizer, loss_func, epoch, perf_file,
          loss_buffer_size=100, batch_size=4, device='cuda',
          prog_bar=None):
    
    perfs = []
    losses = []
    last_test_perf = 0
    best_test_perf = -1
    tot_trials = len(trainX)
    
    permute = np.arange(0, tot_trials)
    np.random.shuffle(permute)
    batches = int(np.ceil(tot_trials / batch_size))
    for batch_idx in range(batches):
        optimizer.zero_grad()
        loss = 0
        for i in range(0, int(min(len(trainX) - (batch_idx*batch_size), 
                              batch_size))
                       ):
            iv = trainX[permute[batch_idx*batch_size + i]]
            
            iv = ttype(iv).unsqueeze(0).unsqueeze(0)
            tv = torch.cuda.LongTensor([trainY[permute[batch_idx*batch_size + i]]])
            out = model(iv)
            loss += loss_func(out[:, -1, :],
                             tv)
            perfs.append((torch.argmax(out[:, -1, :], dim=-1) == 
                      tv).sum().item())
        loss = loss / min(len(trainX) - (batch_idx*batch_size), 
                              batch_size)
        loss.backward()
        optimizer.step()
        
        #perfs.append(0)
        #perfs = perfs[int(-loss_buffer_size/batch_size):]
        losses.append(loss.detach().cpu().numpy())
        losses = losses[-loss_buffer_size:]
        perfs = perfs[-loss_buffer_size:]
        
        if ((batch_idx*batch_size)%loss_buffer_size == 0) & (batch_idx != 0):
            loss_track = {}
            #last_test_perf = test_model(model, 'cuda', test_loader, 
            #                            batch_size)
            loss_track['avg_loss'] = np.mean(losses)
            #loss_track['last_test'] = last_test_perf
            loss_track['training_perf'] = np.mean(perfs)
            loss_track['epoch'] = epoch
            loss_track['batch_idx'] = batch_idx
            with open(perf_file, 'a+') as fp:
                csv_writer = DictWriter(fp, fieldnames=list(loss_track.keys()))
                if fp.tell() == 0:
                    csv_writer.writeheader()
                csv_writer.writerow(loss_track)
                fp.flush()
            if best_test_perf < last_test_perf:
                torch.save(model.state_dict(), perf_file[:-4]+".pt")
                best_test_perf = last_test_perf
        if not (prog_bar is None):
            # Update progress_bar
            s = "{}:{} Loss: {:.5f}, Tperf: {:.4f}, valid: {:.4f}"
            format_list = [epoch,batch_idx*batch_size, np.mean(losses), 
                           np.mean(perfs),
                           last_test_perf]         
            s = s.format(*format_list)
            prog_bar.set_description(s)
            
def test_model(model, device, test_loader, batch_size):
    # Test the Model
    perfs = []
    tot = 0.0
    total_num = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).unsqueeze(1)
            batch = data.shape[0]
            target = target.to(device)
            out = model(data)
            total_num += batch
            tot += (torch.argmax(out[:, -1, :], 
                                 dim=-1) == target).sum().item()
            
    perf = tot / total_num
    return perf

def save_outcome(outcome, filename):
    dat = pd.DataFrame(outcome)
    dat.to_csv(join('perf',filename))

# %% [markdown]
# # Go!

# %%
params = [
          [400, 35, 23, 2],
          ]


# %%
62*16
batch_size=16
loss_buffer_size = 16*62
print(loss_buffer_size)

# %%
#runs = 5
#for r in range(runs):
for i, p in enumerate(params):
    model = gen_model(p)

    tot_weights = 0
    for p in model.parameters():
        tot_weights += p.numel()
    print("Total Weights:", tot_weights)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 10
    Trainscale = 10
    device='cuda'
    progress_bar = tqdm(range(int(epochs)), bar_format='{l_bar}{bar:5}{r_bar}{bar:-5b}')
    epochperfs = []
    times_100 = 0
    for epoch_idx in progress_bar:
        perfs = []
        losses = []
        model.train()
        train(model, ttype, trainX, trainY, None, None,
              optimizer, loss_func, batch_size=batch_size, loss_buffer_size=loss_buffer_size,
              epoch=epoch_idx, perf_file=join('perf','sithcon_audiomixed_startsat1_5052021_sawtooth_{}.csv'.format(0)),
              prog_bar=progress_bar)

# %%
# Load best performing model on training
model.load_state_dict(torch.load('perf/sithcon_audiomixed_startsat1_5052021_sawtooth_0.pt'))

# %%
batch_size = 1
scales = [
          '10.00', 
          '05.00', 
          '02.50',
          '01.25', 
          '01.00', 
          '00.80', 
          '00.40', 
          '00.20', 
          '00.10'
          ]
scale_perf = []
for scale in scales:
    test_paths = join('/media', 'root', 'easystore', 'experiments', 
                     'audiomnist', scale, 'AudioMNIST', 'data')
    full_file = glob.glob(join(test_paths, "*"))
    subj_perfs = []
    for filename in full_file:
        test_dat = np.load(filename)
        
        testX = test_dat[list(all_files.loc[(all_files.rec_split==9)&
                                            (all_files.subjid==int(filename.split("_")[-2])), 'dat_idx'])]
        testY = np.array(all_files.loc[(all_files.rec_split==9)&
                                       (all_files.subjid==int(filename.split("_")[-2])), 'digit'])
        dataset_test = torch.utils.data.TensorDataset(torch.Tensor(testX).cuda(), 
                                                       torch.LongTensor(testY).cuda())
        dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        last_test_perf = test_model(model, 'cuda',
                                    dataset_test, batch_size)
        subj_perfs.append(last_test_perf)
    scoredict = {'perf':np.mean(subj_perfs),
                       'scal':float(scale)}
    print(scoredict)
    scale_perf.append(scoredict)
scale_perfs = pd.DataFrame(scale_perf)
scale_perfs.to_pickle(join("perf", "sith_mixed_test_startsat1_5032021_sawtooth.dill"))

# %%



