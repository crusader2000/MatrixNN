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
from model.matrix_net import *

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

logger = get_logger(conf["para"]["logger_name"])


para = conf["para"]
test_conf = conf["test"]

test_model_path_encoder = test_conf["test_model_path_encoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])
test_model_path_decoder = test_conf["test_model_path_decoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])

# Validate part


def test(enc_model, dec_model, device, snr, Boosting_number):
    model = model.to(device)
    model.eval()
    BER_total = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device).to(torch.float)
            output = data
            for i in range(int(Boosting_number)):
                output = model(output, False)
            results = 1 - torch.sigmoid(output * 1000000)
            bool_equal = (results == target).to(torch.float)
            word_target = conf["data"]["v_size"] * \
                torch.ones(1, conf["para"]["test_batch_size"])
            word_target = word_target.cuda()
            codeword_equal = (torch.sum(bool_equal, -1).cuda()
                              == word_target).to(torch.float)
            BER = 1 - (torch.sum(bool_equal) /
                       (results.shape[0] * results.shape[1]))
            BER_total.append(BER)
        BER = torch.mean(torch.tensor(BER_total))
        snr = para["snr"]
        logger.warning(f"SNR={snr},Boosting_num={int(Boosting_number)-1},BER={BER:.7f}")


if __name__ == "__main__":
    para = conf["para"]
    seed = para["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
		enc_model = MatrixNet(device, conf["data"]["G"]).to(device)
		dec_model = MatrixNet(device, np.transpose(conf["data"]["G"])).to(device)

		enc_model.load_state_dict(torch.load(test_model_path_encoder))
		dec_model.load_state_dict(torch.load(test_model_path_encoder))

    for snr in test_conf["snr_list"].split(","):
        test_conf["snr"] = int(snr)
        for Boosting_number in test_conf["Boosting_number_list"].split(","):
            test(enc_model, dec_model, device, snr, Boosting_number)
