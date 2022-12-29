import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import copy

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
print("available device: {}".format(device))

class BetaAttention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='new', demographic_dim=12, time_aware=False, use_demographic=False):
        super(BetaAttention, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.use_demographic = use_demographic
        self.demographic_dim = demographic_dim
        self.time_aware = time_aware

        if attention_type == 'new':
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))

            self.rate = nn.Parameter(torch.zeros(1)+0.8)
            self.rate2 = nn.Parameter(torch.zeros(1)+0.1)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            
        else:
            raise RuntimeError('Wrong attention type.')
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, input, input2, each_epoch, step, demo=None):
 
        batch_size, time_step, input_dim = input.size() 
        time_decays = (torch.tensor(range(47,-1,-1), dtype=torch.float32)/48).unsqueeze(-1).unsqueeze(0).to(device)
        b_time_decays = time_decays.repeat(batch_size,1,1)+0.01
        
        if self.attention_type == 'new':
            
            q = torch.matmul(input[:,-1,:], self.Wt)
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim)) 
            k = torch.matmul(input, self.Wx)
            dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze() 
            b_value_decays = (torch.sigmoid(input2)/torch.sigmoid(torch.max(abs(input2),1)[0]).unsqueeze(-1)).unsqueeze(-1).unsqueeze(0).to(device)
            f_beta = (1+self.sigmoid(self.rate2))*b_time_decays.squeeze()*b_value_decays.squeeze()/(self.sigmoid(self.rate2)*b_time_decays.squeeze()+b_value_decays.squeeze())
            denominator = (self.rate) * (torch.log(2.72 + (1-self.sigmoid(dot_product))) * (f_beta.squeeze())) * 48
            e = self.relu(self.sigmoid(dot_product)/(denominator))    
        a = self.softmax(e) 
        v = torch.matmul(a.unsqueeze(1), input).squeeze() 

        return v, a

class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', dropout=None):
        super(FinalAttentionQKV, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim


        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1,))
        self.b_out = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2*attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1,))
        
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
 
        batch_size, time_step, input_dim = input.size() 
        input_q = self.W_q(input[:, -1, :]) 
        input_k = self.W_k(input)
        input_v = self.W_v(input)

        if self.attention_type == 'add': 

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim)) 
            h = q + input_k + self.b_in 
            h = self.tanh(h) 
            e = self.W_out(h) 
            e = torch.reshape(e, (batch_size, time_step))

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1)) 
            e = torch.matmul(input_k, q).squeeze()
            
        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1,time_step,1)
            k = input_k
            c = torch.cat((q, k), dim=-1) 
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba 
            e = torch.reshape(e, (batch_size, time_step)) 
        
        a = self.softmax(e) 
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze() 

        return v, a

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index).to(device)

class PositionwiseFeedForward(nn.Module): 
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn 
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) 

        nbatches = query.size(0)
        input_dim = query.size(1)
        feature_dim = query.size(-1)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] 
        
       
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

      
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        DeCov_contexts = x.transpose(0, 1).transpose(1, 2) 
        Covs = cov(DeCov_contexts[0,:,:])
        DeCov_loss = 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 
        for i in range(feature_dim -1 + 1):
            Covs = cov(DeCov_contexts[i+1,:,:])
            DeCov_loss += 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 

        return self.final_linear(x), DeCov_loss

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]) , returned_value[1]
    
class GraphGlobalConv(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphGlobalConv,self).__init__()
        self.hidden_dim = hidden_dim  
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1), 
                                        nn.ReLU(),
                                        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = nn.Sequential(nn.Linear(24*24*128,256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(256,self.hidden_dim))
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,24*24*128)
        x = self.dense(x)
        return x 

class HGV(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model,  MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(HGV, self).__init__()

        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim  
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout = 0, max_len = 400)

        self.LSTMs = clones(nn.LSTM(1, self.hidden_dim, batch_first = True), self.input_dim)
        self.LastStepAttentions = clones(BetaAttention(self.hidden_dim, 32, attention_type='new', demographic_dim=12, time_aware=True, use_demographic=False),self.input_dim)
        
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul',dropout = 1 - self.keep_prob)

        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model,dropout = 1 - self.keep_prob)
        self.SublayerConnection = SublayerConnection(self.d_model, dropout = 1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)

        self.GraphGlobalConv = GraphGlobalConv(self.hidden_dim)
        self.weight1 = nn.Linear(self.hidden_dim, 1)
        self.weight2 = nn.Linear(self.hidden_dim, 1)
        self.global_info_fused = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        self.demo_proj_main = nn.Linear(12, self.hidden_dim)
        self.demo_proj = nn.Linear(12, self.hidden_dim)
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, input, demo_input, graph_data, each_epoch, step):
        gge = self.GraphGlobalConv(graph_data.unsqueeze(1)).unsqueeze(1)
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)
        
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim)
        assert(self.d_model % self.MHD_num_head == 0)
        
        LSTM_embeded_input = self.LSTMs[0](input[:,:,0].unsqueeze(-1), (Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device),Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device)))[0] 
        Attention_embeded_input = self.LastStepAttentions[0](LSTM_embeded_input, input[:,:,0], each_epoch, step)[0].unsqueeze(1)
        for i in range(feature_dim-1):
            embeded_input = self.LSTMs[i+1](input[:,:,i+1].unsqueeze(-1), (Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device),Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device)))[0] 
            embeded_input = self.LastStepAttentions[i+1](embeded_input,input[:,:,i+1], each_epoch, step)[0].unsqueeze(1)
            Attention_embeded_input = torch.cat((Attention_embeded_input, embeded_input), 1)

#         weight1 = torch.sigmoid(self.weight1(gge))
#         weight2 = torch.sigmoid(self.weight2(demo_main))
#         weight1 = weight1/(weight1+weight2)
#         weight2 = 1 - weight1
#         global_info_embedd = weight1 * gge + weight2 * demo_main
        
        global_info_embedd = self.global_info_fused(torch.cat((gge, demo_main), 2))

        Attention_embeded_input = torch.cat((Attention_embeded_input, global_info_embedd), 1)
        posi_input = self.dropout(Attention_embeded_input) 

        contexts = self.SublayerConnection(posi_input, lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None))
    
        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(contexts, lambda x: self.PositionwiseFeedForward(contexts))[0]
        weighted_contexts = self.FinalAttentionQKV(contexts)[0]
        output = self.output1(self.relu(self.output0(weighted_contexts)))
        output = self.sigmoid(output)
          
        return output, DeCov_loss
