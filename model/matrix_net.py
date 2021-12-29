import torch
import torch.nn as nn
import numpy as np


class MatrixNet(nn.Module):
    def __init__(self, device, matrix):
        super(MatrixNet, self).__init__()
        self.clip_tanh = 10
        self.num_edges = 0
        self.matrix = matrix
        self.k,self.n = len(matrix),len(matrix[0])
        self.device = device
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

                    if "v"+str(i) not in self.adjancency_list:
                        self.adjancency_list["v"+str(i)] = [("c"+str(j),self.num_edges)]
                    else:
                        self.adjancency_list["v"+str(i)].append(("c"+str(j),self.num_edges))
                    
                    if "c"+str(j) not in self.adjancency_list:
                        self.adjancency_list["c"+str(j)] = [("v"+str(i),self.num_edges)]
                    else:
                        self.adjancency_list["c"+str(j)].append(("v"+str(i),self.num_edges))
                    
                    self.edges[self.num_edges] = ("v"+str(i),"c"+str(j))
                    self.num_edges = self.num_edges + 1

        self.input_layer_mask = torch.zeros(self.k, self.num_edges).to(device)
        self.output_layer_mask = torch.zeros(self.num_edges, self.n).to(device)
        self.odd_to_even_layer_mask = torch.zeros(self.num_edges, self.num_edges).to(device)
        self.even_to_odd_layer_mask = torch.zeros(self.num_edges, self.num_edges).to(device)

        self.zero_indices = torch.nonzero(1-self.even_to_odd_layer_mask)
        
        for i in range(self.num_edges):
            self.input_layer_mask[int(self.edges[i][0][1]),i] = 1

        for i in range(self.n):
            for _,e_num in self.adjancency_list["c"+str(i)]:
                self.output_layer_mask[e_num,i] = 1
    
        for i in range(self.num_edges):
            for _,e_num in self.adjancency_list[self.edges[i][1]]: 
                self.odd_to_even_layer_mask[e_num,i] = 1
            for _,e_num in self.adjancency_list[self.edges[i][0]]: 
                self.even_to_odd_layer_mask[i,e_num] = 1

        self.odd_to_even_layer_mask = (self.odd_to_even_layer_mask - torch.eye(self.num_edges).to(device))
        self.even_to_odd_layer_mask = (self.even_to_odd_layer_mask - torch.eye(self.num_edges).to(device))

        self.weights_odd1_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd1_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_odd2_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd2_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_odd3_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd3_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_output_we = nn.Parameter(torch.randn(self.num_edges, self.n))

    def odd_layer(self, inputs_v, inputs_e, oddw_v, oddw_e):
        inputs_v = inputs_v.to(torch.float)
        v_out = torch.mul(inputs_v, oddw_v).to(torch.float)
        inputs_e = inputs_e.to(torch.float)
        e_out = torch.matmul(inputs_e, torch.mul(self.odd_to_even_layer_mask,oddw_e).to(torch.float)).to(torch.float)

        odd = v_out + e_out
        odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
        odd = torch.tanh(odd)
        return odd

    def even_layer(self, odd, flag_clip):
        num_m,_ = odd.size()
        odd_repeat = torch.zeros(num_m,self.num_edges,self.num_edges).to(self.device)

        # print(odd.size())
        # print(odd_repeat.size())
        for i in range(num_m):
            odd_repeat[i,:,:] = torch.reshape(odd[i].repeat(1,self.num_edges),(self.num_edges,self.num_edges)).to(torch.float)

        # print(odd_repeat.size())
        # odd_repeat.to(self.device)
        # self.even_to_odd_layer_mask.to(self.device)

        # Check Following Line
        # Try torch.contract if doesnt work
        print(self.even_to_odd_layer_mask)
        even = torch.mul(odd_repeat,self.even_to_odd_layer_mask).to(torch.float)
        print(even)

        for i in range(num_m):
            even[i][self.zero_indices] = 1
        print(even)
        
        # print(even.size())
        
        prod_rows = torch.prod(even,dim = 1,keepdim=False).to(torch.float)
        print(prod_rows.size())
        # prod_rows = prod_rows.resize(1,self.num_edges)
        print(prod_rows)

        if flag_clip:
            prod_rows = torch.clamp(prod_rows, min=-self.clip_tanh, max=self.clip_tanh)
        
        prod_rows = torch.log(torch.div(1 + prod_rows, 1 - prod_rows))
        return prod_rows

    def output_layer(self, inputs_e, oddw_e):
        inputs_e = inputs_e.to(torch.float)
        print(inputs_e.size())
        # print(torch.mul(self.odd_to_even_layer_mask,oddw_e).size())
        e_out = torch.matmul(inputs_e, torch.mul(self.output_layer_mask,oddw_e)).to(torch.float)
        
        # o_c = torch.special.expit(e_out)
        # o_c = (o_c > 0.5).to(torch.int)
        return e_out

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

        output = self.output_layer(odd_result, self.weights_output_we)

        return output
