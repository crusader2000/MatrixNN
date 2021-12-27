#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.utils.data
from data_loader import *
from IPython import display

import pickle
import glob
import os
import logging
import time
from datetime import datetime
from ast import literal_eval
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
from PIL import Image

import reed_muller_modules
from reed_muller_modules.logging_utils import *

from opt_einsum import contract   # This is for faster torch.einsum
from reed_muller_modules.reedmuller_codebook import *
from reed_muller_modules.hadamard import *
from reed_muller_modules.comm_utils import *
from reed_muller_modules.logging_utils import *
from reed_muller_modules.all_functions import *
# import reed_muller_modules.reedmuller_codebook as reedmuller_codebook

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

#python train_KO_m2.py --gpu 0 --m 8 --enc_train_snr 0 --dec_train_snr -2 --batch_size 10000 --small_batch_size 250encoder0

parser = argparse.ArgumentParser(description='(m,2) dumer')

parser.add_argument('--m', type=int, default=8, help='reed muller code parameter m')

parser.add_argument('--batch_size', type=int, default=50000, help='size of the batches')

parser.add_argument('--small_batch_size', type=int, default=25000, help='size of the batches')

parser.add_argument('--hidden_size', type=int, default=32, help='neural network size')

parser.add_argument('--full_iterations', type=int, default=20000, help='full iterations')
parser.add_argument('--enc_train_iters', type=int, default=50, help='encoder iterations')
parser.add_argument('--dec_train_iters', type=int, default=500, help='decoder iterations')

parser.add_argument('--enc_train_snr', type=float, default=0., help='snr at enc are trained')
parser.add_argument('--dec_train_snr', type=float, default=-2., help='snr at dec are trained')



parser.add_argument('--loss_type', type=str, default='BCE', choices=['MSE', 'BCE'], help='loss function')

parser.add_argument('--gpu', type=int, default=7, help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()

device = torch.device("cuda:{0}".format(args.gpu))
kwargs = {'num_workers': 4, 'pin_memory': False}

results_save_path = './Results/RM({0},2)/fullNN_Enc+fullNN_Dec/Enc_snr_{1}_Dec_snr{2}/Batch_{3}'\
    .format(args.m, args.enc_train_snr,args.dec_train_snr, args.batch_size)
os.makedirs(results_save_path, exist_ok=True)
os.makedirs(results_save_path+'/Models', exist_ok = True)



def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

# LSE (Log Sum Exponential) is used for decoding RM(m,1) codewords
def log_sum_exp(LLR_vector):

    sum_vector = LLR_vector.sum(dim=1, keepdim=True)
    sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)

    return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1) 

# Calculating BER
def errors_ber(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return res


# Calculating BLER
def errors_bler(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate

# Function used for calculating the norm of the output of neural networks
def power_constraint(codewords, gnet_top, power_constraint_type, training_mode):
    return F.normalize(codewords, p=2, dim=1)*np.sqrt(2**args.m)

# More operations on LLR bits
def llr_info_bits(hadamard_transform_llr, order_of_RM1):

    max_1, _ = hadamard_transform_llr.max(1, keepdim=True)
    min_1, _ = hadamard_transform_llr.min(1, keepdim=True)

    LLR_zero_column = max_1 + min_1 

    
    max_zero, _ = torch.max(contract('ij, kj ->  ikj', hadamard_transform_llr.abs() , first_order_Mul_Ind_Zero_dict[order_of_RM1]), 2)
    max_one, _ =  torch.max(contract('ij, kj ->  ikj', hadamard_transform_llr.abs() , first_order_Mul_Ind_One_dict[order_of_RM1]), 2)

    LLR_remaining = max_zero - max_one

    return torch.cat([LLR_zero_column, LLR_remaining], dim=1)


def modified_llr_codeword(LLR_Info_bits, order_of_RM1):

    required_LLR_info = contract('ij , jk ->ikj', LLR_Info_bits, first_order_generator_dict[order_of_RM1]) 

    sign_matrix = (-1)**((required_LLR_info < 0).sum(2)).float() 

    min_abs_LLR_info, _= torch.min(torch.where(required_LLR_info==0., torch.max(required_LLR_info.abs())+1, required_LLR_info.abs()), dim = 2)

    return sign_matrix * min_abs_LLR_info

# RM(2,2) decoder using Soft MAP
def RM_22_SoftMAP_decoder(LLR):

    
    max_PlusOne, _ = torch.max(contract('lk, ijk ->  lij', LLR, RM_22_codebook_PlusOne), 2)
    max_MinusOne, _ =  torch.max(contract('lk, ijk ->  lij', LLR, RM_22_codebook_MinusOne), 2)
    
    return max_PlusOne - max_MinusOne


# Leaves are Reed Muller codes


def awgn_channel(codewords, snr):
    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords)
    corrupted_codewords = codewords+noise_sigma * standard_Gaussian
    return corrupted_codewords


############################

def get_msg_bits_batch(data_generator):
    msg_bits_batch = next(data_generator)
    return msg_bits_batch

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


bers = []
losses = []
codebook_size = 1000

def pairwise_distances(codebook):
    dists = []
    for row1, row2 in combinations(codebook, 2): 
        distance = (row1-row2).pow(2).sum()
        dists.append(np.sqrt(distance.item()))
    return dists, np.min(dists)

# Training Algorithm
try:
    for k in range(args.full_iterations):
        start_time = time.time()
        msg_bits_large_batch = 2 * (torch.rand(args.batch_size, code_dimension_k) < 0.5).float() - 1

        num_small_batches = int(args.batch_size/args.small_batch_size)
        #     # Train decoder  
        for _ in range(args.dec_train_iters):
            dec_optimizer.zero_grad()        
            for i in range(num_small_batches):
                start, end = i*args.small_batch_size, (i+1)*args.small_batch_size
                msg_bits = msg_bits_large_batch[start:end].to(device)
                transmit_codewords = correct_second_order_encoder_Neural_RM_leaves(msg_bits, gnet_dict)      
                corrupted_codewords = awgn_channel(transmit_codewords, args.dec_train_snr)
                decoded_bits = correct_second_order_decoder_nn_full(corrupted_codewords, fnet_dict)

                loss = criterion(decoded_bits,  0.5*msg_bits+0.5)/num_small_batches
                
                loss.backward()
            dec_optimizer.step()
            
                
        # Train Encoder
        for _ in range(args.enc_train_iters):

            enc_optimizer.zero_grad()        

            for i in range(num_small_batches):
                start, end = i*args.small_batch_size, (i+1)*args.small_batch_size
                msg_bits = msg_bits_large_batch[start:end].to(device)
            
                transmit_codewords = correct_second_order_encoder_Neural_RM_leaves(msg_bits, gnet_dict)       
                corrupted_codewords = awgn_channel(transmit_codewords, args.enc_train_snr)
                decoded_bits = correct_second_order_decoder_nn_full(corrupted_codewords, fnet_dict)       

                loss = criterion(decoded_bits, 0.5*msg_bits+0.5 )/num_small_batches
                
                loss.backward()
            
            enc_optimizer.step()
            
            ber = errors_ber(msg_bits, decoded_bits.sign()).item()
            
        bers.append(ber)

        losses.append(loss.item())
        if k % 10 == 0:
            print('[%d/%d] At %d dB, Loss: %.10f BER: %.10f' 
                % (k+1, args.full_iterations, args.enc_train_snr, loss.item(), ber))
            print("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))


        # Save the model for safety
        if k % 10 == 0:

            torch.save(dict(zip(['g{0}'.format(i) for i in range(3, args.m+1)], [gnet_dict[i].state_dict() for i in range(3, args.m+1)])),\
                    results_save_path+'/Models/Encoder_NN_{0}.pt'.format(k+1))

            torch.save(dict(zip(['f{0}'.format(i) for i in range(1,2*args.m-3)], [fnet_dict[i].state_dict() for i in range(1, 2*args.m-3)])),\
                    results_save_path+'/Models/Decoder_NN_{0}.pt'.format(k+1))

            plt.figure()
            plt.plot(bers)
            plt.plot(moving_average(bers, n=10))
            plt.savefig(results_save_path +'/training_ber.png')
            plt.close()

            plt.figure()
            plt.plot(losses)
            plt.plot(moving_average(losses, n=10))
            plt.savefig(results_save_path +'/training_losses.png')
            plt.close()

except KeyboardInterrupt:
    print('Graceful Exit')
else:
    print('Finished')

plt.figure()
plt.plot(bers)
plt.plot(moving_average(bers, n=10))
plt.savefig(results_save_path +'/training_ber.png')
plt.close()

plt.figure()
plt.plot(losses)
plt.plot(moving_average(losses, n=10))
plt.savefig(results_save_path +'/training_losses.png')
plt.close()

torch.save(dict(zip(['g{0}'.format(i) for i in range(3, args.m+1)], [gnet_dict[i].state_dict() for i in range(3, args.m+1)])),\
                    results_save_path+'/Models/Encoder_NN.pt')

torch.save(dict(zip(['f{0}'.format(i) for i in range(1,2*args.m-3)], [fnet_dict[i].state_dict() for i in range(1, 2*args.m-3)])),\
                    results_save_path+'/Models/Decoder_NN.pt')


