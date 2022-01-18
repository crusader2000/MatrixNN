#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.utils.data
from IPython import display

from util.conf_util import *
from util.log_util import *
from util.utils import *
from model.matrix_net_v2 import *

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

	n = len(conf["data"]["G"])
	m = len(conf["data"]["G"][0])

	enc_model = MatrixNet(device, conf["data"]["G"][:,n:]).to(device)
	dec_model = MatrixNet(device, np.transpose(conf["data"]["G"])).to(device)
	start_epoch = 0

	if para["retrain"]:
		train_model_path_encoder = para["train_save_path_encoder"].format(para["retrain_day"],para["data_type"],para["retrain_epoch_num"])
		train_model_path_decoder = para["train_save_path_decoder"].format(para["retrain_day"],para["data_type"],para["retrain_epoch_num"])
		enc_model.load_state_dict(torch.load(train_model_path_encoder))
		dec_model.load_state_dict(torch.load(train_model_path_decoder))
		start_epoch = int(para["retrain_epoch_num"])
		logger.info("Retraining Model " + conf_name + " : " +str(para["retrain_day"]) +" Epoch: "+str(para["retrain_epoch_num"]))


	criterion = BCEWithLogitsLoss()
	enc_optimizer = optim.ADAM(enc_model.parameters(), lr=para["lr"])
	dec_optimizer = optim.ADAM(dec_model.parameters(), lr=para["lr"])
	enc_scheduler = ReduceLROnPlateau(enc_optimizer, 'min')
	dec_scheduler = ReduceLROnPlateau(dec_optimizer, 'min')

	data_type = para["data_type"]
	
	bers = []
	losses = []
	
	train_save_dirpath = para["train_save_path_dir"].format(today, data_type)
	if not os.path.exists(train_save_dirpath):
		os.makedirs(train_save_dirpath)
	
	torch.autograd.set_detect_anomaly(True)

	# Training Algorithm
	try:
		for k in range(start_epoch, para["full_iterations"]):
			start_time = time.time()
			msg_bits_large_batch = 2*torch.randint(0,2,(para["train_batch_size"], para["k"])).to(torch.float) -1

			num_small_batches = int(para["train_batch_size"]/para["train_small_batch_size"])

			# Train decoder  
			for iter_num in range(para["dec_train_iters"]):
				dec_optimizer.zero_grad()        
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_bits = msg_bits_large_batch[start:end].to(device)
					codewords = enc_model(msg_bits)      
					# print("codewords")
					# print(codewords)
					transmit_codewords = F.normalize(torch.hstack((msg_bits,codewords)), p=2, dim=1)*np.sqrt(2**para["m"])
					# print("transmit_codewords")
					# print(transmit_codewords)
					corrupted_codewords = awgn_channel(transmit_codewords, para["dec_train_snr"])
					# print("corrupted_codewords")
					# print(corrupted_codewords)
					
					decoded_bits = dec_model(corrupted_codewords)
					# print("decoded_bits")
					# decoded_bits = torch.nan_to_num(decoded_bits,0.0)
					# print(decoded_bits)
					loss = criterion(decoded_bits, msg_bits)/num_small_batches
					
					# print(loss)
					loss.backward()
				dec_scheduler.step(loss)
				print("Decoder",iter_num)
				dec_optimizer.step()
					
			# Train Encoder
			for iter_num in range(para["enc_train_iters"]):
				enc_optimizer.zero_grad()        
				ber = 0
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_bits = msg_bits_large_batch[start:end].to(device)						
					codewords = enc_model(msg_bits)      
					transmit_codewords = F.normalize(torch.hstack((msg_bits,codewords)), p=2, dim=1)*np.sqrt(2**para["m"])
					corrupted_codewords = awgn_channel(transmit_codewords, para["enc_train_snr"])
					decoded_bits = dec_model(corrupted_codewords)

					loss = criterion(decoded_bits, msg_bits )/num_small_batches
					
					loss.backward()
					ber += errors_ber(msg_bits, decoded_bits.sign()).item()

				enc_scheduler.step(loss)
				print("Encoder",iter_num)
				enc_optimizer.step()
				ber /= num_small_batches	
				
			bers.append(ber)
			logger.info('[%d/%d] At ENC SNR %f dB DEC SNR %f dB, Loss: %.10f BER: %.10f' 
							% (k+1, para["full_iterations"], para["enc_train_snr"], para["dec_train_snr"], loss.item(), ber))
			logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

			losses.append(loss.item())
			if k % 20 == 0:
				# Save the model for safety
				torch.save(enc_model.state_dict(), para["train_save_path_encoder"].format(today, data_type, k+1))
				torch.save(dec_model.state_dict(), para["train_save_path_decoder"].format(today, data_type, k+1))

				plt.figure()
				plt.plot(bers)
				plt.plot(moving_average(bers, n=10))
				plt.legend(("bers","moving_average"))
				plt.xlabel("Iterations")
				plt.savefig(train_save_dirpath +'/training_ber.png')
				plt.close()

				plt.figure()
				plt.plot(losses)
				plt.plot(moving_average(losses, n=10))
				plt.legend(("bers","moving_average"))
				plt.xlabel("Iterations")
				plt.savefig(train_save_dirpath +'/training_losses.png')
				plt.close()

	except KeyboardInterrupt:
		logger.warning('Graceful Exit')
		exit()
	else:
		logger.warning('Finished')

	plt.figure()
	plt.plot(bers)
	plt.plot(moving_average(bers, n=5))
	plt.legend(("bers","moving_average"))
	plt.xlabel("Iterations")

	plt.savefig(train_save_dirpath +'/training_ber.png')
	plt.close()

	plt.figure()
	plt.plot(losses)
	plt.plot(moving_average(losses, n=5))
	plt.legend(("bers","moving_average"))
	plt.xlabel("Iterations")
	plt.savefig(train_save_dirpath +'/training_losses.png')
	plt.close()

	torch.save(enc_model.state_dict(), para["train_save_path_encoder"].format(today, data_type, para["full_iterations"]))
	torch.save(dec_model.state_dict(), para["train_save_path_decoder"].format(today, data_type, para["full_iterations"]))
	
