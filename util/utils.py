import torch
import numpy as np

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
	corrupted_codewords = codewords+ (noise_sigma/2)*standard_Gaussian
	return corrupted_codewords

