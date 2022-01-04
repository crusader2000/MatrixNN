import yaml
import numpy as np
from .reed_muller import *
import scipy
from scipy.linalg import lu

def get_default_conf(conf_path=None):
    if conf_path is None:
        conf_path = "./config/default.yaml"
    with open(conf_path, "r") as f_conf:
        conf = yaml.load(f_conf.read(), Loader=yaml.FullLoader)

    data_type = conf["para"]["data_type"]

    m = conf["para"]["m"]
    r = conf["para"]["r"]

#    conf["para"]["test_model_path"] = conf["para"]["test_model_path"].format(
 #       data_type, data_type)
    conf["para"]["logger_name"] = conf["para"]["logger_name"].format(data_type)
    
    G = get_gen_matrix(m,r)
    k = len(G)
    n = len(G[0])
    print(G)
    print("Num Edges Before",np.sum(np.sum(G)))
    p,l,u = lu(G)
    # row_reduced_G = Matrix(G).rref()
    # row_reduced_G = np.matrix(Matrix(G).rref())
    print(l)
    print("Num Edges After",np.sum(np.sum(l)))


    rate = 1.0 * k / n

    conf["para"]["n"] = n
    conf["para"]["k"] = k
    conf["data"]["rate"] = rate

    conf["data"]["G"] = G
    
    return conf
