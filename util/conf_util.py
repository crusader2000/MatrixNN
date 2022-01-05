import yaml
import numpy as np
from .reed_muller import *
# import scipy
# from scipy.linalg import lu

def GaussElim(mat):
    n = len(mat)
    m = len(mat[0])
    lead = 0
    for r in range(n):
        if lead >= m:
            break
        
        i = r

        while mat[i,lead] == 0:
            i = i+1
            if i == n:
                i = r
                lead += 1
                if lead >= m:
                    break

        if i != r:
            temp = np.copy(mat[i,:])
            mat[i,:] = np.copy(mat[r,:])
            mat[r,:] = np.copy(temp)

        for i in range(n):
            if i != r and mat[i,lead] == 1:
                mat[i,:] ^= mat[r,:]
        lead += 1

    curr = 0
    for i in range(m):
        if curr == n:
            break
        if np.sum(mat[:,i]) == 1 :
            temp = np.copy(mat[:,i])
            mat[:,i] = np.copy(mat[:,curr])
            mat[:,curr] = np.copy(temp)
            curr += 1
    
    return mat 


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
    
    G = GaussElim(G)
    # p,l,u = lu(G)
    # row_reduced_G = Matrix(G).rref()
    # row_reduced_G = np.matrix(Matrix(G).rref())
    print(G)
    print("Num Edges After",np.sum(np.sum(G)))


    rate = 1.0 * k / n

    conf["para"]["n"] = n
    conf["para"]["k"] = k
    conf["data"]["rate"] = rate

    conf["data"]["G"] = G
    
    return conf
