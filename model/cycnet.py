import torch
import torch.nn as nn
import numpy as np


class cycnet(nn.Module):
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

        

        # self.oddw_v1 = nn.Parameter(torch.randn(1, self.num_edges))
        # self.oddw_e1 = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        # self.oddw_v2 = nn.Parameter(torch.randn(1, self.num_edges))
        # self.oddw_e2 = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        # self.oddw_v3 = nn.Parameter(torch.randn(1, self.num_edges))
        # self.oddw_e3 = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        # self.oddw_v4 = nn.Parameter(torch.randn(1, self.num_edges))
        # self.oddw_e4 = nn.Parameter(torch.randn(self.num_edges, self.num_edges))
        # self.oddw_v5 = nn.Parameter(torch.randn(1, self.num_edges))
        # self.oddw_e5 = nn.Parameter(torch.randn(self.num_edges, self.num_edges))


    # def odd_layer(self, inputs_v, inputs_e, oddw_v, oddw_e):
    #     inputs_v = inputs_v.to(torch.float)
    #     inputs_v = inputs_v.unsqueeze(2)
    #     # batch_size * v_size * l_size = (batch_size * v_size * 1) * ( 1 * l )
    #     v_out = torch.matmul(inputs_v, oddw_v)
    #     # inputs_v count by column  b*e = b*v*l
    #     v_out = v_out.reshape(-1, self.e_size)

    #     # To do cycrow to cyccolumn: b * e_size = (b * e_size) * (e_size * e*size)
    #     inputs_e = torch.matmul(inputs_e.to(
    #         torch.float), self.permutations_cycrowtocyccol.to(torch.float))
    #     # b*e = b*v*l * l*l
    #     mask_w_e = torch.mul(oddw_e, self.mask_e)
    #     inputs_e = inputs_e.view(-1, self.v_size, self.l_size,).to(torch.float)
    #     e_out = torch.matmul(inputs_e, mask_w_e)
    #     e_out = e_out.view(-1, self.e_size)

    #     # add v_out and e_out
    #     odd = v_out + e_out
    #     odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
    #     odd = torch.tanh(odd)
    #     return odd

    # def even_layer(self, odd, flag_clip):
    #     # To do column to row
    #     even = torch.matmul(odd.to(torch.float),
    #                         self.permutations_cyccoltocycrow.to(torch.float))
    #     # Cumulative product then divide itself
    #     even = even.view(-1, self.v_size, self.l_size)
    #     # Matrix value:0->1
    #     even = torch.add(even, 1 - (torch.abs(even) > 0).to(torch.float))
    #     prod_even = torch.prod(even, -1)
    #     even = torch.div(prod_even.unsqueeze(2).repeat(
    #         1, 1, self.l_size), even).reshape(-1, self.e_size)
    #     if flag_clip:
    #         even = torch.clamp(even, min=-self.clip_tanh, max=self.clip_tanh)
    #     even = torch.log(torch.div(1 + even, 1 - even))
    #     return even

    # def output_layer(self, inputs_v, inputs_e):
    #     out_layer1 = torch.matmul(inputs_e.to(
    #         torch.float), self.permutations_cycrowtocyccol.to(torch.float))
    #     out_layer2 = out_layer1.to(torch.float)
    #     # b*v = (b*e) * (e*v)
    #     out_layer3 = out_layer2.view(-1, self.v_size, self.l_size)
    #     # b*v = (b*v*l) * (l)
    #     e_out = torch.matmul(out_layer3, self.w_e_out)
    #     v_out = inputs_v.to(torch.float)
    #     return v_out + e_out

    # def forward(self, x, is_train=True):
    #     flag_clip = 1
    #     if is_train:
    #         message = self.train_message
    #     else:
    #         message = self.test_message
    #     odd_result = self.odd_layer(x, message, self.oddw_v1, self.oddw_e1)
    #     even_result1 = self.even_layer(odd_result, flag_clip)

    #     flag_clip = 0
    #     odd_result = self.odd_layer(
    #         x, even_result1, self.oddw_v2, self.oddw_e2)
    #     even_result2 = self.even_layer(odd_result, flag_clip)

    #     odd_result = self.odd_layer(
    #         x, even_result2, self.oddw_v3, self.oddw_e3)
    #     even_result3 = self.even_layer(odd_result, flag_clip)

    #     odd_result = self.odd_layer(
    #         x, even_result3, self.oddw_v4, self.oddw_e4)
    #     even_result4 = self.even_layer(odd_result, flag_clip)

    #     odd_result = self.odd_layer(
    #         x, even_result4, self.oddw_v5, self.oddw_e5)
    #     even_result5 = self.even_layer(odd_result, flag_clip)

    #     output = self.output_layer(x, even_result5)

    #     return output
