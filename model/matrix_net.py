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

        # print("self.odd_to_even_layer_mask")
        # print(self.odd_to_even_layer_mask)
        # print(torch.sum(self.odd_to_even_layer_mask,dim=1,keepdim=False))
        # print("self.even_to_odd_layer_mask")
        # print(self.even_to_odd_layer_mask)
        # print(torch.sum(self.even_to_odd_layer_mask,dim=1,keepdim=False))

        self.zero_indices = ((self.even_to_odd_layer_mask == 0)).nonzero()

        self.weights_odd1_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd1_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_odd2_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd2_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_odd3_wv = nn.Parameter(torch.randn(1, self.num_edges))
        self.weights_odd3_we = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        self.weights_output_we = nn.Parameter(torch.randn(self.num_edges, self.n))

    def odd_layer(self, inputs_v, inputs_e, oddw_v, oddw_e):
        # inputs_v = torch.nan_to_num(inputs_v,0.0,posinf=float('inf')-10,neginf=-float('inf')+10)
        # inputs_e = torch.nan_to_num(inputs_e,0.0,posinf=float('inf')-10,neginf=-float('inf')+10)
        
        inputs_v = inputs_v.to(torch.float)
        v_out = torch.mul(inputs_v, oddw_v).to(torch.float)
        inputs_e = inputs_e.to(torch.float)
        e_out = torch.matmul(inputs_e, torch.mul(self.odd_to_even_layer_mask,oddw_e).to(torch.float)).to(torch.float)

        odd = v_out + e_out
        odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
        odd = torch.tanh(odd)
        return odd

    def even_layer(self, odd, flag_clip):
        # odd = torch.nan_to_num(odd,0.0,posinf=float('inf')-10,neginf=-float('inf')+10)
        
        num_m,_ = odd.size()
        odd_repeat = torch.zeros(num_m,self.num_edges,self.num_edges).to(self.device)
        # print("odd")
        # print(odd)
        # print(odd.size())
        # print(odd_repeat.size())
        for i in range(num_m):
            odd_repeat[i,:,:] = torch.reshape(odd[i].repeat(1,self.num_edges),(self.num_edges,self.num_edges)).to(torch.float)
        # print("odd_repeat")
        # print(odd_repeat)
        # print(odd_repeat.size())
        
        # print(odd_repeat.size())
        # odd_repeat.to(self.device)
        # self.even_to_odd_layer_mask.to(self.device)

        # Check Following Line
        # Try torch.contract if doesnt work
        # print("self.even_to_odd_layer_mask")
        # print(self.even_to_odd_layer_mask)
        even = torch.mul(odd_repeat,self.even_to_odd_layer_mask).to(torch.float)
        # print("even")
        # print(even)
        # print("self.zero_indices")
        # print(self.zero_indices)
        
        even = torch.add(even,1-self.even_to_odd_layer_mask)
        
        # for i in range(num_m):
        #     even[i] = torch.add(even[i],1-self.even_to_odd_layer_mask)
        
            # print(even[i])
            # for x,y in self.zero_indices:
            #     even[i,x,y] = 1
            # print(even[i])

        # print("even")
        # print(even)
        
        # print(even.size())
        
        prod_rows = torch.prod(even,dim = 1,keepdim=False).to(torch.float)
        # print("prod_rows.size()")
        # print(prod_rows.size())
        # prod_rows = prod_rows.resize(1,self.num_edges)
        # print("prod_rows")
        # print(prod_rows)

        if flag_clip:
            prod_rows = torch.clamp(prod_rows, min=-self.clip_tanh, max=self.clip_tanh)
        
        # print("prod_rows")
        # print(prod_rows)
        
        # prod_rows_clone = prod_rows.clone()
        # prod_rows[prod_rows_clone==1] = 0.99999
        # prod_rows[prod_rows_clone==-1] = -0.9999

        prod_rows = torch.log(torch.div(1 + prod_rows, 1 - prod_rows))
        # print(prod_rows)
        
        
        return prod_rows

    def output_layer(self, inputs_e, oddw_e):
        # inputs_e = torch.nan_to_num(inputs_e,0.0,posinf=float('inf')-10,neginf=-float('inf')+10)

        inputs_e = inputs_e.to(torch.float)
        # print(inputs_e.size())
        # print(torch.mul(self.odd_to_even_layer_mask,oddw_e).size())
        e_out = torch.matmul(inputs_e, torch.mul(self.output_layer_mask,oddw_e)).to(torch.float)
        
        # o_c = torch.special.expit(e_out)
        # o_c = (o_c > 0.5).to(torch.int)
        return e_out

    def forward(self, x):
        x = x.to(torch.float)

        flag_clip = 1
        lv = torch.matmul(x,self.input_layer_mask).to(torch.float)
        # print("lv")
        # print(lv)
        odd_result = self.odd_layer(lv, lv, self.weights_odd1_wv, self.weights_odd1_we)
        # print("odd_result")
        # print(odd_result)
        even_result1 = self.even_layer(odd_result, flag_clip)
        # print(even_result1)

        flag_clip = 1
        odd_result = self.odd_layer(lv, even_result1, self.weights_odd2_wv, self.weights_odd2_we)
        # print("odd_result")
        # print(odd_result)
        even_result2 = self.even_layer(odd_result, flag_clip)
        # print(even_result2)

        odd_result = self.odd_layer(lv, even_result2, self.weights_odd3_wv, self.weights_odd3_we)
        # print("odd_result")
        # print(odd_result)
        even_result3 = self.even_layer(odd_result, flag_clip)
        # print(even_result3)

        output = self.output_layer(odd_result, self.weights_output_we)
        # print(output)

        return output
