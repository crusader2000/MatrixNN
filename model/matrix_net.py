import torch
import torch.nn as nn
import numpy as np


class MatrixNet(nn.Module):
    def __init__(self, conf, device):
        super(cycnet, self).__init__()
        self.clip_tanh = 10
        # self.v_size = conf["data"]['v_size']
        # self.e_size = conf["data"]['e_size']
        self.num_edges = 0

        self.k,self.n = np.size(self.matrix)

        ############################
        # Matrix : 
        #   c1 c2 ..... cn
        # v1
        # v2
        # v3
        # .
        # .
        # .
        # vk
        #

        self.edges = {}
        self.adjancency_list = {}

        for i in range(self.k):
            for j in range(self.n):
                if self.matrix[i,j] == 1:

                    if not self.adjancency_list["v"+str(i)]:
                        self.adjancency_list["v"+str(i)] = [("c"+str(j),self.num_edges)]
                    else:
                        self.adjancency_list["v"+str(i)].append(("c"+str(j),self.num_edges))
                    
                    if not self.adjancency_list["c"+str(j)]:
                        self.adjancency_list["c"+str(j)] = [("v"+str(i),self.num_edges)]
                    else:
                        self.adjancency_list["c"+str(j)].append(("v"+str(i),self.num_edges))
                    
                    self.edges[self.num_edges] = ("v"+str(i),"c"+str(j))
                    self.num_edges = self.num_edges + 1

        self.input_layer_mask = torch.zeros(self.k, self.num_edges).to(device)
        self.output_layer_mask = torch.zeros(self.num_edges, self.n).to(device)
        self.odd_to_even_layer_mask = torch.zeros(self.num_edges, self.num_edges).to(device)
        self.even_to_odd_layer_mask = torch.zeros(self.num_edges, self.num_edges).to(device)

        for i in range(self.num_edges):
            self.input_layer_mask[self.edges[i][0],i] = 1

        for i in range(self.n):
            for _,e_num in self.adjancency_list["c"+str(i)]:
                self.output_layer_mask[e_num,i] = 1
    
        for i in range(self.num_edges):
            for _,e_num in self.adjancency_list[self.edges[i][1]]: 
                self.odd_to_even_layer_mask[e_num,i] = 1
            for _,e_num in self.adjancency_list[self.edges[i][0]]: 
                self.even_to_odd_layer_mask[e_num,i] = 1

        self.odd_to_even_layer_mask = (self.odd_to_even_layer_mask - torch.eye(self.num_edges)).to(device)  
        self.even_to_odd_layer_mask = (self.even_to_odd_layer_mask - torch.eye(self.num_edges)).to(device)  

        self.weights_odd1_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd1_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_odd2_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd2_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_odd3_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd3_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_output_we = nn.Parameter(torch.randn(self.num_edges, self.n))

    def odd_layer(self, inputs_v, inputs_e, oddw_v, oddw_e):
        inputs_v = inputs_v.to(torch.float)
        v_out = torch.mul(inputs_v, oddw_v,dtype=torch.float)

        inputs_e = inputs_e.to(torch.float)
        e_out = torch.matmul(inputs_e, torch.mul(self.odd_to_even_layer_mask,oddw_e,dtype=torch.float),dtype=torch.float)

        odd = v_out + e_out
        odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
        odd = torch.tanh(odd)
        return odd

    def even_layer(self, odd, flag_clip):
        even = odd.repeat(self.num_edges,dtype=torch.float)
        even = torch.matmul(even,self.even_to_odd_layer_mask,dtype=torch.float)

        even[torch.nonzero(1-self.even_to_odd_layer_mask)] = 1

        even = torch.prod(even,dim = self.num_edges,keepdim=False,dtype=torch.float)

        if flag_clip:
            even = torch.clamp(even, min=-self.clip_tanh, max=self.clip_tanh)
        even = torch.log(torch.div(1 + even, 1 - even))
        return even

    def output_layer(self, inputs_e, oddw_e):
        inputs_e = inputs_e.to(torch.float)
        e_out = torch.matmul(inputs_e, torch.mul(self.odd_to_even_layer_mask,oddw_e),dtype=torch.float)
        
        o_c = torch.special.expit(e_out)
        o_c = (o_c > 0.5).to(torch.int)
        return o_c

    def forward(self, x):
        x = x.to(torch.float)

        flag_clip = 1
        lv = torch.matmul(x,self.input_layer_mask).to(torch.float)

        odd_result = self.odd_layer(lv, lv, self.weights_odd1_wv, self.weights_odd1_we)
        even_result1 = self.even_layer(odd_result, flag_clip)

        flag_clip = 0
        odd_result = self.odd_layer(lv, even_result1, self.weights_odd2_wv, self.weights_odd2_we)
        even_result2 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(lv, even_result2, self.weights_odd3_wv, self.weights_odd3_we)
        even_result3 = self.even_layer(odd_result, flag_clip)

        output = self.output_layer(odd_result, weights_output_we)

        return output
