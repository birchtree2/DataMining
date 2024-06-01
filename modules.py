import math
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from torch.distributions import Normal

def add(x, y):
    return x+y

class OnehotEmbedding(nn.Module):
    def __init__(self, word_number):
        super().__init__()
        self.word_number = word_number
    
    def one_hot_encoding(self, labels):
        one_hot = torch.zeros(labels.size(0), labels.size(1), self.word_number).cuda()
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return one_hot
        
    def forward(self, x):
        embeddings = F.one_hot(x, self.word_number).float()[:,:,1:]
        return embeddings

class Attention(nn.Module):
    def __init__(self, hidden, alignment_network='dot'):
        super().__init__()
        self.style = alignment_network.lower()
        if self.style == 'general':
            self.transform = nn.Linear(hidden, hidden)
        elif self.style == 'bilinear':
            self.weight = nn.Parameter(torch.Tensor(hidden, hidden))
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        
    def forward(self, query, key):
        if self.style == 'dot':
            return torch.bmm(query, 
                             key.transpose(1, 2))
        elif self.style == 'general':
            return torch.bmm(query, 
                             self.transform(key).transpose(1, 2))
        elif self.style == 'bilinear':
            # return self.transform(query, key).squeeze(-1)
            return torch.bmm(query.matmul(self.weight), 
                             key.transpose(1, 2))
        elif self.style == 'decomposable':
            return torch.bmm(self.transform(query),
                             self.transform(key).transpose(1, 2))

class CNN_Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.seq_len = 512
        self.d_model = 24
        nhead = 8  
        num_layers = 6  
        channel = 8
        self.embedding = OnehotEmbedding(21)
        self.conv1 = nn.Conv1d(20, self.d_model, 3, padding=1)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=32,  # 前馈神经网络中的隐藏层维度
            dropout=0.1,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.convs = nn.Sequential(
            nn.Conv1d(1, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2)
        )
        total_layers=9
        for i in range(total_layers - 1):
            self.convs.add_module('conv{}'.format(i), nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False))
            self.convs.add_module('pool{}'.format(i), nn.AvgPool1d(2))
        self.mlp=nn.Linear(channel*self.d_model,128)

    def forward(self, sequence, masks):
        batch_size=sequence.shape[0]
        x = self.embedding(sequence)
        x = x.permute(0,2,1) #[B,K,L]
        # x = x.contiguous().view(-1, 1, self.seq_len) #[B*K,1,L]
        x = self.conv1(x) 
        x = x.permute(0,2,1) #[B,L,K]
        x = self.encoder(x) #[B,L,d]
        x = x.contiguous().view(-1, 1, self.seq_len)
        #[B*d,1,L]
        x = self.convs(x)
        x=x.squeeze(-1) #[B*d,channel]
        x = x.view(batch_size, self.d_model, -1) #[B,d,channel]
        x = x.view(batch_size, -1) #[B,d*channel]
        x = self.mlp(x)
        return x
        

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        channel = 8
        self.input_size = 20
        self.max_seq_length = 512
        total_layers = int(math.log2(self.max_seq_length))
        self.convs = nn.Sequential(
            nn.Conv1d(1, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )
        for i in range(total_layers - 1):
            self.convs.add_module('conv{}'.format(i), nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False))
            self.convs.add_module('pool{}'.format(i), nn.AvgPool1d(2))
  
    def forward(self, x, masks):
        seq_num = len(x) #[B,L,K] B:batch_size, L:seq_length(512), K:num_amino_acids(20)
        x = x.permute(0, 2, 1) #[B,K,L]
        
        x = x.contiguous().view(-1, 1, self.max_seq_length) #[B*K,1,L]
        x = self.convs(x) #[B*K,CHANNEL,1]
        x=x.squeeze(-1) #[B*K,channel]
        x = x.view(seq_num, self.input_size, -1) #[B,K,channel]
        
        return x
    
# 
        

class RNN(nn.Module):
    def __init__(self, input_size=-1, hidden_size=-1, num_layers=1, output_size=None, rnn_type='lstm', bidirectional=True, dropout=0.0):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if bidirectional:
            hidden_size //= 2
            
        self.output_size = output_size if output_size is not None else hidden_size

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("Invalid rnn_type. Choose either 'lstm' or 'gru'.")

    def forward(self, x, masks):
        """
        self:
            x: (batch_size, seq_len, input_size)
            masks: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, output_size)
            hidden: (batch_size, num_layers * num_directions, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, masks.sum(1).int().cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=seq_len)
        return output
          


class Model(nn.Module):
    def __init__(self, hidden_size, 
                 embedding_type="onehot",
                 encoder_type="cnn"):
        super().__init__()
        if embedding_type == "onehot":
            self.embedding = OnehotEmbedding(21)
            hidden_size = 8
        elif embedding_type == "embedding":
            self.embedding = nn.Embedding(21, hidden_size, padding_idx=0)
        if encoder_type == "cnn":
            self.encoder = CNN()
            self.rnn_encoder = RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        elif encoder_type == "rnn":
            self.encoder = RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        # elif encoder_type == "transformer":
        #     self.encoder=nn.TransformerEncoderLayer()
        self.output_layer = nn.Linear(160, 128)
    
    def forward(self, seq_inputs, seq_masks):
        batch_size = seq_inputs.shape[0] # [B, 512]
        
        sequences = self.embedding(seq_inputs) #[B, 512, 20]
        
        sequence_embeddings = self.encoder(sequences, seq_masks) #[B,20,8]
        
        sequence_embeddings = sequence_embeddings.view(batch_size, -1) # [B, 160]

        final_embedding = self.output_layer(sequence_embeddings) #[B,128]

        return final_embedding


import copy
class GCNNModel(nn.Module):
    def __init__(self, hidden_size, 
                 embedding_type="onehot",
                 encoder_type="cnn"):
        super().__init__()
        self.emb_dim=20 #20 amino acids
        self.embedding = OnehotEmbedding(self.emb_dim+1)
        self.seq_len=512
        self.hidden_dim = 32
        kernel_size = 9
        dilation = 2
        self.num_cnn_layers=3
        self.linear1 = nn.Conv1d(self.emb_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.fc=nn.Linear(self.seq_len,self.seq_len)
        self.convs = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        # nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        # nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        # nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        # nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)]
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, dilation=dilation**1, padding=(kernel_size-1)//2*(dilation**1)),
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, dilation=dilation**2, padding=(kernel_size-1)//2*(dilation**2)),
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, dilation=dilation**3, padding=(kernel_size-1)//2*(dilation**3))]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(self.num_cnn_layers)])
        #把上面的复制num遍
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for i in range(self.num_cnn_layers)])
        #self.gate和self.convs结构一样
        self.gates = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(self.num_cnn_layers)])
        self.linear2=nn.Conv1d(self.hidden_dim, self.hidden_dim//2, kernel_size=kernel_size, padding=4)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim*self.seq_len//2, 4096),
                                          nn.GELU(),
                                          nn.Linear(4096,4096),
                                        #   nn.LayerNorm(4096),
                                        #   nn.Linear(4096,1024),
                                          nn.GELU(),
                                          nn.Linear(4096, 128),
                                          nn.LayerNorm(128))
    def forward(self, seq_inputs, seq_masks):
        batch_size = seq_inputs.shape[0] # [B, 512]
        input_size = seq_inputs.shape[1]
        sequences = self.embedding(seq_inputs) #[B, 512, 20]
        
        x=sequences.permute(0,2,1) #[B,20,512]
        x=self.fc(x)
        x=self.linear1(x)
        x=F.gelu(x) #[B,64,512]
        for i in range(self.num_cnn_layers):
            t=self.norms[i](x.permute(0,2,1)) #h.shape [B,512,64]
            h=F.gelu(self.convs[i](t.permute(0,2,1)))
            g=F.sigmoid(self.gates[i](t.permute(0,2,1)))
            x=h*g+x
        # x:[B,64,512]
        x=self.linear2(x) #[B,32,512]
        x=x.view(batch_size, -1) #[B,32*512]
        x=self.mlp(x)
        return x
        #use nn.linear
        

    #     def __init__(self,din=20,dk=32,dv=32):
#         super().__init__()
#         self.din=din
#         self.dk=dk
#         self.dv=dv
#         self.embedding = OnehotEmbedding(self.din+1)
#         self.wq=nn.Linear(din,dk)
#         self.wk=nn.Linear(din,dk)
#         self.wv=nn.Linear(din,dv)
#         self.norm_fact=1/math.sqrt(dk)
#         self.max_seq_length = 512
#         total_layers = int(math.log2(self.max_seq_length))
#         self.channel = 8
#         self.convs = nn.Sequential(
#             nn.Conv1d(1, self.channel, 3, 1, padding=1, bias=False),
#             nn.AvgPool1d(2),
#         )
#         for i in range(total_layers - 1):
#             self.convs.add_module('conv{}'.format(i), nn.Conv1d(self.channel, self.channel, 3, 1, padding=1, bias=False))
#             self.convs.add_module('pool{}'.format(i), nn.AvgPool1d(2))

        
#     def forward(self, seq_inputs, seq_masks):
#         batchsize=seq_inputs.shape[0]
#         x=self.embedding(seq_inputs) #[B,512,20]
#         #[B,L,K] B:batch_size, L:seq_length(512), K:num_amino_acids(20)
#         Q=self.wq(x)
#         K=self.wk(x)
#         K_T=K.permute(0,2,1) #[B,20,512]
#         V=self.wv(x)
#         A= nn.Softmax(dim=-1)(torch.bmm(Q, K_T) * self.norm_fact) #[B,512,512]
#         x = torch.bmm(A, V) #[B,512,16]
#         x = x.contiguous().view(-1, 1, self.max_seq_length) #[B*dv,1,L]
#         x = self.convs(x) #[B*dv,CHANNEL,1]
#         x=x.squeeze(-1) #[B*dv,channel]
#         x = x.view(batchsize, -1)
#         return x