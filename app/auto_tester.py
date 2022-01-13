import random
import os
import sys

import numpy as np
import torch
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from util.conf_util import *
from util.log_util import *
from util.utils import *
from model.matrix_net import *

import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date

if len(sys.argv) == 2:
    conf_name = sys.argv[1]
    print("test conf_name:", conf_name)
    conf = get_default_conf(f"./config/{conf_name}.yaml")
else:
    print("default")
    conf = get_default_conf()

if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
else:
    device = torch.device("cpu")


para = conf["para"]
test_conf = conf["test"]

today = date.today().strftime("%b-%d-%Y")

logger = get_logger(test_conf["logger_name"])
logger.info("test_conf_name : "+conf_name)
logger.info("Device : "+str(device))

test_size = para["test_size"]
test_model_path_encoder = test_conf["test_model_path_encoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])
test_model_path_decoder = test_conf["test_model_path_decoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])

test_save_dirpath = para["train_save_path_dir"].format(test_conf["day"], para["data_type"])
if not os.path.exists(test_save_dirpath):
    os.makedirs(test_save_dirpath)

# Validate part
def test(enc_model, dec_model, device, snr):
    enc_model = enc_model.to(device)
    enc_model.eval()
    dec_model = dec_model.to(device)
    dec_model.eval()
    
    BER_total = []
    Test_msg_bits = 2*torch.randint(0,2,(test_size, para["k"])).to(torch.float) -1
    Test_Data_Generator = DataLoader(Test_msg_bits, batch_size=100 , shuffle=False)

    num_test_batches = len(Test_Data_Generator)
    ber = 0
    start_time = time.time()

    with torch.no_grad():
        for msg_bits in Test_Data_Generator:
            msg_bits = msg_bits.to(device)
            codewords = enc_model(msg_bits)      
            transmit_codewords = F.normalize(torch.hstack((msg_bits,codewords)), p=2, dim=1)*np.sqrt(2**para["m"])
            corrupted_codewords = awgn_channel(transmit_codewords, snr)
            decoded_bits = dec_model(corrupted_codewords)
            ber += errors_ber(msg_bits, decoded_bits.sign()).item()

        ber /= num_test_batches
        logger.warning(f"[Testing Block] SNR={snr} : BER={ber:.7f}")
        logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

    return ber

if __name__ == "__main__":
    logger.info("Testing code!!!!")
    para = conf["para"]
    seed = para["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(conf["data"]["G"])
    m = len(conf["data"]["G"][0])

    enc_model = MatrixNet(device, conf["data"]["G"][:,n:]).to(device)
    dec_model = MatrixNet(device, np.transpose(conf["data"]["G"])).to(device)

    enc_model.load_state_dict(torch.load(test_model_path_encoder))
    dec_model.load_state_dict(torch.load(test_model_path_decoder))
    
    bers = []
    snrs = []
    logger.info("Testing {} trained till epoch_num {}".format(conf_name,test_conf["epoch_num"]))
    logger.info("Model trained on {}".format(test_conf["day"]))
    logger.info("Less go!")
    for snr in test_conf["snr_list"].split(","):
        ber = test(enc_model, dec_model, device, int(snr))
        bers.append(ber)
        snrs.append(int(snr))
    
    plt.plot(snrs, bers, label=" ",linewidth=2, color='blue')

    plt.xlabel("SNRs")
    plt.ylabel("BERs (Testing)")
    plt.title("Testing BERs")
    plt.savefig(test_save_dirpath+ "/ber_testing_epoch_"+str(test_conf["epoch_num"])+".png")
    plt.close()
