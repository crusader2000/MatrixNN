#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.utils.data
from IPython import display

from util.conf_util import *
from util.log_util import *
from model.matrix_net import *

import sys
import pickle
import glob
import os
import logging
import time
from datetime import datetime
from datetime import date
import random


import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

from opt_einsum import contract   # This is for faster torch.einsum

from model.matrix_net import MatrixNet


def snr_db2sigma(train_snr):
		return 10**(-train_snr*1.0/20)

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

def awgn_channel(codewords, snr):
		noise_sigma = snr_db2sigma(snr)
		standard_Gaussian = torch.randn_like(codewords)
		corrupted_codewords = codewords+noise_sigma * standard_Gaussian
		return corrupted_codewords


############################

def moving_average(a, n=3) :
		ret = np.cumsum(a, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return ret[n - 1:] / n


bers = []
losses = []

if __name__ == "__main__":
	if len(sys.argv) == 2:
		conf_name = sys.argv[1]
		print("train conf_name:", conf_name)
		conf = get_default_conf(f"./config/{conf_name}.yaml")
	else:
		print("default")
		conf = get_default_conf()

	if torch.cuda.is_available():
			device = torch.device("cuda")
			os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
			print(device,os.environ["CUDA_VISIBLE_DEVICES"])
	else:
			device = torch.device("cpu")
			print(device)
	

	para = conf["para"]
	seed = para["seed"]
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	today = date.today().strftime("%b-%d-%Y")

	logger = get_logger(para["logger_name"])
	logger.info("train_conf_name : "+conf_name)
	logger.info("Device : "+str(device))
	logger.info("We are on!!!")

	enc_model = MatrixNet(device, conf["data"]["G"]).to(device)
	dec_model = MatrixNet(device, np.transpose(conf["data"]["G"])).to(device)
	criterion = BCEWithLogitsLoss()
	enc_optimizer = optim.RMSprop(enc_model.parameters(), lr=para["lr"])
	dec_optimizer = optim.RMSprop(dec_model.parameters(), lr=para["lr"])
	
	data_type = para["data_type"]
	
	bers = []
	losses = []

	# Training Algorithm
	try:
			for k in range(para["full_iterations"]):
					start_time = time.time()
					msg_bits_large_batch = 2*torch.randint(0,2,(para["train_batch_size"], para["k"])) -1

					num_small_batches = int(para["train_batch_size"]/para["train_small_batch_size"])

					# Train decoder  
					for _ in range(para["dec_train_iters"]):
							dec_optimizer.zero_grad()        
							for i in range(num_small_batches):
									start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
									msg_bits = msg_bits_large_batch[start:end].to(device)
									codewords = enc_model(msg_bits)      
									transmit_codewords = F.normalize(codewords, p=2, dim=1)*np.sqrt(2**para["m"])
									corrupted_codewords = awgn_channel(transmit_codewords, para["snr"])
									decoded_bits = dec_model(corrupted_codewords)

									loss = criterion(decoded_bits, 0.5*msg_bits+0.5)/num_small_batches
									
									print(i)
									loss.backward()
							dec_optimizer.step()
							
					# Train Encoder
					for _ in range(para["enc_train_iters"]):

							enc_optimizer.zero_grad()        

							for i in range(num_small_batches):
								start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
								msg_bits = msg_bits_large_batch[start:end].to(device)
						
								codewords = enc_model(msg_bits)      
								transmit_codewords = F.normalize(codewords, p=2, dim=1)*np.sqrt(2**para["m"])      
								corrupted_codewords = awgn_channel(transmit_codewords, para["snr"])
								decoded_bits = dec_model(corrupted_codewords)

								loss = criterion(decoded_bits, msg_bits )/num_small_batches
								
								loss.backward()

							enc_optimizer.step()
							
							ber = errors_ber(msg_bits, decoded_bits.sign()).item()
							
					bers.append(ber)
					logger.info('[%d/%d] At %d dB, Loss: %.10f BER: %.10f' 
									% (k+1, para["full_iterations"], para["enc_train_snr"], loss.item(), ber))
					logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

					losses.append(loss.item())
					if k % 10 == 0:
							train_save_path_encoder = para["train_save_path_decoder"].format(today, data_type, k+1)
							if not os.path.exists(train_save_path_encoder):
								os.makedirs(train_save_path_encoder)
							
							train_save_path_decoder = para["train_save_path_decoder"].format(today, data_type, k+1)
							if not os.path.exists(train_save_path_decoder):
									os.makedirs(train_save_path_decoder)

							

							# Save the model for safety
							torch.save(enc_model.state_dict(), para["train_save_path_encoder"].format(today, data_type, k+1))
							torch.save(dec_model.state_dict(), para["train_save_path_decoder"].format(today, data_type, k+1))

							plt.figure()
							plt.plot(bers)
							plt.plot(moving_average(bers, n=10))
							plt.savefig(para["train_save_path_decoder"].format(today, data_type, k+1) +'/training_ber.png')
							plt.close()

							plt.figure()
							plt.plot(losses)
							plt.plot(moving_average(losses, n=10))
							plt.savefig(para["train_save_path_decoder"].format(today, data_type, k+1) +'/training_losses.png')
							plt.close()

	except KeyboardInterrupt:
			logger.warning('Graceful Exit')
	else:
			logger.warning('Finished')

	train_save_path_encoder = para["train_save_path_decoder"].format(today, data_type, para["full_iterations"])
	if not os.path.exists(train_save_path_encoder):
		os.makedirs(train_save_path_encoder)
	
	train_save_path_decoder = para["train_save_path_decoder"].format(today, data_type, para["full_iterations"])
	if not os.path.exists(train_save_path_decoder):
			os.makedirs(train_save_path_decoder)

	plt.figure()
	plt.plot(bers)
	plt.plot(moving_average(bers, n=10))
	plt.savefig(para["train_save_path_decoder"].format(today, data_type, para["full_iterations"]) +'/training_ber.png')
	plt.close()

	plt.figure()
	plt.plot(losses)
	plt.plot(moving_average(losses, n=10))
	plt.savefig(para["train_save_path_decoder"].format(today, data_type, para["full_iterations"]) +'/training_losses.png')
	plt.close()

	torch.save(enc_model.state_dict(), para["train_save_path_encoder"].format(today, data_type, para["full_iterations"]))
	torch.save(dec_model.state_dict(), para["train_save_path_decoder"].format(today, data_type, para["full_iterations"]))
	