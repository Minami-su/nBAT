import os
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import f1_score
from torchtext import data, datasets
import pandas as pd
import pickle
from sklearn import metrics
from torchtext.vocab import GloVe, CharNGram
from numba import jit
from apex import amp
from Warmup import adjust_learning_rate
from mlp import Model
from sklearn.metrics import confusion_matrix
base_dir = os.path.abspath(os.path.join(os.getcwd()))
from sklearn.metrics import matthews_corrcoef
atis_data = os.path.join(base_dir, 'data')
from sklearn.metrics import roc_auc_score
import torch.nn.functional as Fn
'''
build train and val dataset
'''

tokenize = lambda s: s.split()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
SOURCE = data.Field(sequential=True, tokenize=tokenize,
                    lower=True, use_vocab=True,
                    init_token='[SOS]',
                    pad_token='<pad>', unk_token='mask',
                    batch_first=True, fix_length=2160,
                    include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence
YUFA = data.Field(sequential=True, tokenize=tokenize,
                    lower=True, use_vocab=True,
                    init_token='[SOS]',
                    pad_token='<pad>', unk_token='<unk>',
                    batch_first=True, fix_length=555,
                    include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence

LABEL = data.Field(
    sequential=False,unk_token=None,
    use_vocab=True)

train, val, test = data.TabularDataset.splits(
    path=atis_data,
    skip_header=True,
    train='biosyu3L.train.csv',
    validation='biosyu3L.valid.csv',
    test='biosyu3L.test.csv',
    format='csv',
    fields=[('source', SOURCE), ('target', LABEL), ('yufa', YUFA)])

SOURCE.build_vocab(train, val, test)
LABEL.build_vocab(train, val, test)
YUFA.build_vocab(train, val, test)
train_iter, val_iter,test_iter = data.Iterator.splits(
    (train, val,test),
    batch_sizes=(128,128,128),  # 训练集设置为32,验证集整个集合用于测试
    shuffle=False,
    sort_within_batch=True,  # 为true则一个batch内的数据会按sort_key规则降序排序
    sort_key=lambda x: len(x.source))  # 这里按src的长度降序排序，主要是为后面pack,pad操作)

# # save source words
# source_words_path = os.path.join(os.getcwd(), 'source_words.pkl')
# with open(source_words_path, 'wb') as f_source_words:
#     pickle.dump(SOURCE.vocab, f_source_words)

# # save label words
# label_words_path = os.path.join(os.getcwd(), 'label_words.pkl')
# with open(label_words_path, 'wb') as f_label_words:
#     pickle.dump(LABEL.vocab, f_label_words)
    


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math
import time
import torch.nn.functional as Fn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class pooling(nn.Module):
    def __init__(self):
        super(pooling, self).__init__()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, ceil_mode=False)
    def forward(self, input):
        return self.maxpool1(input)
class pooling2(nn.Module):
    def __init__(self):
        super(pooling2, self).__init__()
        self.avgpool2 = nn.AvgPool1d(kernel_size=15, ceil_mode=False)
    def forward(self, input):
        return self.avgpool2(input)
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, src_pad_idx, max_length=2200):
        super(Encoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.src_pad_idx= src_pad_idx
        cache_dir = './'
        cache_dir2 = './'

        self.word_embeddings = nn.Embedding(input_dim,hid_dim, padding_idx=self.src_pad_idx)
        self.word_embeddings.weight.data.uniform_(-1., 1.)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        #self.w_q = nn.Linear(200, hid_dim).to(device)
        #self.w_q2 = nn.Linear(85, hid_dim).to(device)
        # 多层encoder
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.pool = pooling()
        self.pool2 = pooling2()
        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)
    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
    def forward(self, src, src_mask,yu):
        # src:[batch_size, src_len]
        # src_mask:[batch_size, 1, 1, src_len]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src2=src.clone()
        src2=src2.float()
        src2=self.pool2(src2).to(device)
        src2=self.pool(src2).to(device)
        src_mask = self.make_src_mask(src2)
        # pad_mask = (src == 1).to(DEVICE)
        # y=torch.tensor(-float('inf')).to(DEVICE)
        # pad_mask,y, src=pad_mask,y, src.float()
        # src=torch.where(pad_mask,y, src).to(DEVICE)
        src_mask =None
        # inf_count = torch.sum(torch.eq(src, -float('inf'))).item()
        # print(inf_count)
        #src=src.transpose(0,1).to(DEVICE)
        #print(src)
        #src=src.view(src.size(0),1,src.size(1),src.size(2)).to(device)
        
        #print(src.shape)
        #src=src.transpose(0,1).to(DEVICE)
        #print(src)
        # inf_count = torch.sum(torch.eq(src, -float('inf'))).item()
        # print(inf_count)
        # src=src.long()
        x = self.word_embeddings(src).to(device)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        #emb = torch.cat((x1, x2), dim=2).to(device)
        
        # inf_count = torch.sum(torch.eq(emb, 0)).item()
        # print(inf_count)
        #x = self.w_q(emb)
        # 位置信息
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # token编码+位置编码
        src = self.dropout(x + self.pos_embedding(pos))
        #print(src)
        src=src.transpose(1,2).to(DEVICE)
        src=self.pool2(src).to(device)
        src=self.pool(src).to(device)
        #src=src.view(src.size(0),src.size(2),src.size(3)).to(device)
        #src=self.w_q2(src).to(device)
        src=src.transpose(1,2).to(DEVICE)
        i = 0
        kl=0
        for layer in self.layers:
            if i == 0:
                
                src, biaW = layer(src, src_mask)
                pos = src
            else:
                src, biaW2 = layer(src, src_mask)  # [batch_size, src_len, hid_dim]
                #kl = Syntacticdependency(yu, biaW2)
            i += 1
        return src,kl


@jit(nopython=True)
def cadoo(in_data):
    x = 0
    pe = np.zeros((len(in_data),len(in_data[0]),len(in_data[0])))
    for m in in_data:
        t = len(m)
        τ = 0.1
        e = -1e9
        k = 0
        for i in m:
            l = 0
            for j in m:
                if l <= k:
                    pe[x][k][l] = -abs(i - j) / τ
                else:
                    pe[x][k][l] = e
                l += 1
            k += 1
        x += 1
    return pe

def Syntacticdependency(in_data,biaffineW):
    #start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_data=in_data.cpu()
    in_data=np.array(in_data,dtype=np.float32)
    pe3=cadoo(in_data)
    #print("time: ", time.time() - start_time)
    pe3 = torch.tensor(pe3).to(device)
    #pe3 = pe3.to(device)
    pe3=pe3.view(-1,pe3.size(-1),pe3.size(-1))
    #print(pe3.shape)

    pe3=Fn.softmax(pe3,dim=-1)
    e = -1e9
    T = pe3.size(-1)
    
    t1 = biaffineW.view(pe3.size(0),T,T)

    t1 = t1[:, 1:, 1:]
    pe3 = pe3[:, 1:, 1:]

    t1 = Fn.log_softmax(t1, dim=-1)
    

    kl = (1 / T) * Fn.kl_div(t1.float(),pe3.float(), reduction='batchmean')

    
    return kl

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src:[batch_size, src_len, hid_dim]
        # src_mask:[batch_size, 1, 1, src_len]

        # 1.经过多头attetnion后，再经过add+norm
        # self-attention
        _src, biaW = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))  # [batch_size, src_len, hid_dim]

        # 2.经过一个前馈网络后，再经过add+norm
        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))  # [batch_size, src_len, hid_dim]

        return src, biaW

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query:[batch_size, query_len, hid_dim]
        # key:[batch_size, query_len, hid_dim]
        # value:[batch_size, query_len, hid_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch_size, query_len, n_heads, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # [batch_size, n_heads, query_len, key_len]
        
        if mask is not None:
            energy = energy.mask_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1) # [batch_size, n_heads, query_len, key_len]
        
        x = torch.matmul(self.dropout(attention), V) # [batch_size, n_heads, query_len, head_dim]
        
        x = x.permute(0, 2, 1, 3).contiguous() # [batch_size, query_len, n_heads, head_dim]
        
        x = x.view(batch_size, -1, self.hid_dim) # [batch_size, query_len, hid_dim]
        
        x = self.fc_o(x) # [batch_size, query_len, hid_dim]
        
        return x,energy
# class MultiHeadAttentionLayer(nn.Module):
#     def __init__(self, hid_dim, n_heads, dropout):
#         super(MultiHeadAttentionLayer, self).__init__()
#         assert hid_dim % n_heads == 0

#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#         self.head_dim = hid_dim // n_heads

#         self.fc_q = nn.Linear(hid_dim, hid_dim)
#         self.fc_k = nn.Linear(hid_dim, hid_dim)
#         self.fc_v = nn.Linear(hid_dim, hid_dim)

#         self.fc_o = nn.Linear(hid_dim, hid_dim)

#         self.dropout = nn.Dropout(dropout)

#         self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.shape[0]

#         # query:[batch_size, query_len, hid_dim]
#         # key:[batch_size, query_len, hid_dim]
#         # value:[batch_size, query_len, hid_dim]

#         Q = self.fc_q(query)
#         K = self.fc_k(key)
#         V = self.fc_v(value)

#         Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
#                                                                         3)  # [batch_size, query_len, n_heads, head_dim]
#         K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         Q1 = Q[:, 3:, :, :]
#         K1 = K[:, 3:, :, :]
#         V1 = V[:, 3:, :, :]
#         Q = Q[:, 0:3, :, :]
#         K = K[:, 0:3, :, :]
#         V = V[:, 0:3, :, :]

#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]

#         if mask is not None:
#             energy = energy.mask_fill(mask == 0, -1e10)

#         attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, query_len, key_len]
        
#         x = torch.matmul(self.dropout(attention), V)  # [batch_size, n_heads, query_len, head_dim]

#         x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
#         # print(x.shape)
#         # x = x.view(batch_size,3, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         weight = nn.Parameter(torch.randn(batch_size, Q1.size(1), Q1.size(3), Q1.size(3))).to(device)
#         Q1 = Q1 @ weight
#         energy1 = torch.matmul(Q1, K1.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]

#         if mask is not None:
#             energy1 = energy1.mask_fill(mask == 0, -1e10)

#         attention1 = torch.softmax(energy1, dim=-1)  # [batch_size, n_heads, query_len, key_len]
#         # print(attention1.shape)
#         x1 = torch.matmul(self.dropout(attention1), V1)  # [batch_size, n_heads, query_len, head_dim]

#         x1 = x1.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]

#         # x1 = x1.view(batch_size, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]
#         x = torch.cat([x, x1], dim=2)
#         x = x.view(batch_size, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]
#         x = self.fc_o(x)  # [batch_size, query_len, hid_dim]

#         return x, energy1


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x:[batch_size, seq_len, hid_dim]

        x = self.dropout(self.gelu(self.fc_1(x)))  # [batch_size, seq_len, pf_dim]
        x = self.fc_2(x)  # [batch_size, seq_len, hid_dim]

        return x


class BERT(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout,intent_size, slot_size, src_pad_idx):
        super(BERT, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, src_pad_idx)
        self.gelu = nn.GELU()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Dropout(dropout), nn.Tanh(),nn.LayerNorm(hid_dim))
        self.intent_out = nn.Linear(hid_dim, intent_size)
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.mlp= Model(hid_dim,hid_dim*4, slot_size).to(device)

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]

    def forward(self, src,yu):
        #src_mask = self.make_src_mask(src)
        src_mask = 1
        encoder_out,kl = self.encoder(src, src_mask,yu)  # [batch_size, src_len, hid_dim]

        # 拿到[cls] token进行意图分类
        cls_hidden = self.fc(encoder_out[:, 0])  # [batch_size, hid_dim]
        intent_output = self.intent_out(cls_hidden)  # [batch_size, intent_size]
        intent_output = Fn.softmax(intent_output, dim=-1)

        return intent_output,kl


n_layers =2  # transformer-encoder层数
n_heads = 4  # 多头self-attention
hid_dim = 768
dropout = 0
pf_dim = 768* 4

input_dim = len(SOURCE.vocab)
intent_size = len(LABEL.vocab)
print(input_dim)
print(intent_size)# intent size
src_pad_idx = SOURCE.vocab.stoi[SOURCE.pad_token]
# for i in range(input_dim):
#     lv = SOURCE.vocab.itos[i]
#     print(lv)
# for i in range(input_dim2):
#     lv = SOURCEM.vocab.itos[i]
#     print(lv)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = BERT(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, intent_size,input_dim, src_pad_idx).to(DEVICE)
model4=model

print(f'The model has {count_parameters(model):,} trainable parameters')
# 优化函数
optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01)

# 损失函数(slot)
loss_slot = nn.CrossEntropyLoss()
# ignore_index=src_pad_idx
# 定义损失函数(意图识别)
loss_intent = nn.CrossEntropyLoss()

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
# def randomMask(x):
#     x = x.clone()
#     # number of elements to replace in each row
#     num_elements_to_replace = int(x.shape[1] * 0.25)
#     # create a tensor to store mlm_positions
#     mlm_positions = torch.zeros((x.shape[0], num_elements_to_replace), dtype=torch.long)

#     for i in range(x.shape[0]):
#         # randomly select indices to replace with 1
#         indices_to_replace = torch.randperm(x.shape[1])[:num_elements_to_replace]
#         # replace the selected indices with 1
#         x[i][indices_to_replace] = 0
#     #print(x)
#     one_indices = torch.where(x == 0)
#     # Get the indices of the elements in the second dimension
#     mlm_positions = one_indices[1].view(x.shape[0], -1)
#     mlm_positions = mlm_positions
#     #print(mlm_positions)
#     return x,mlm_positions
# def randomMask(x):
#     x = x.clone()
#     # find the number of non-1 elements
#     non_one_elements = torch.sum(x != 1, dim=1)
#     # number of elements to replace in each row
#     num_elements_to_replace = (non_one_elements * 0.15).long()
#     one_indices = torch.where(x == 0)
#     mlm_positions = []
#     for i in range(x.shape[0]):
#         # find the indices of non-1 elements
#         non_one_indices = torch.where(x[i] != 1)[0]
#         # randomly select indices to replace among non-1 elements
#         indices_to_replace = torch.randperm(non_one_indices.shape[0])[:num_elements_to_replace[i]]
#         indices_to_replace = non_one_indices[indices_to_replace]
#         one_indices_i = torch.where(x[i] == 0)[0]
#         one_indices_i = one_indices_i[indices_to_replace]
#         mlm_positions.append(one_indices_i)
#     mlm_positions = torch.stack(mlm_positions, dim=0)
#     return x,mlm_positions
# def encodermlm(X,mlm_positions):
#     pred_positions=mlm_positions
#     num_pred_positions = pred_positions.shape[1]
#     pred_positions = pred_positions.reshape(-1)
#     batch_size = X.shape[0]
#     batch_idx = torch.arange(0, batch_size)
#     # 假设batch_size=2，num_pred_positions=3
#     # 那么batch_idx是np.array（[0,0,0,1,1,1]）
#     batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
#     masked_X = X[batch_idx, pred_positions]
#     masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
#     #print(masked_X)  # tensor([1, 2])
#     #print(masked_X.shape)
#     return masked_X
# def euclidean_distance(tensor1, tensor2):
#     # Subtract the tensors element-wise
#     diff = tensor1 - tensor2
#     # Square the element-wise differences
#     sq_diff = diff**2
#     # Sum the element-wise differences
#     sum_sq_diff = sq_diff.sum()
#     # Take the square root of the sum
#     distance = sum_sq_diff.sqrt()
#     return distance

# def chayi(tensor1,result):

#     distance = euclidean_distance(tensor1,result)
#     print(distance)

#     comparison = torch.ne(tensor1, result)

#     # 计算值为1的个数
#     count = comparison.sum().item()

#     # 计算张量中总值的个数
#     total = tensor1.numel()

#     # 计算差异概率
#     difference_probability = count / total

#     print(1-difference_probability)  # 输出0

# class MaskLM(nn.Module):
#     """BERT的掩蔽语言模型任务"""
#     def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
#         super(MaskLM, self).__init__(**kwargs)
#         self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
#                                  nn.ReLU(),
#                                  nn.LayerNorm(num_hiddens),
#                                  nn.Linear(num_hiddens, vocab_size)).to(DEVICE)

#     def forward(self, X, pred_positions):
#         num_pred_positions = pred_positions.shape[1]
#         pred_positions = pred_positions.reshape(-1)
#         batch_size = X.shape[0]
#         batch_idx = torch.arange(0, batch_size)
#         # 假设batch_size=2，num_pred_positions=3
#         # 那么batch_idx是np.array（[0,0,0,1,1,1]）
#         batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
#         masked_X = X[batch_idx, pred_positions]
#         masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
#         masked_X = masked_X.to(DEVICE)
#         mlm_Y_hat = self.mlp(masked_X).to(DEVICE)
#         mlm_Y_hat = Fn.softmax(mlm_Y_hat, dim=-1)
#         return mlm_Y_hat

def train(model, iterator, optimizer,  loss_intent, clip):
    start_time = time.time()
    model.train()
    epoch_loss = 0

    #scaler = GradScaler()
    for i, batch in enumerate(iterator):
        src, _ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
        label = batch.target
        yu, _ = batch.yufa
        src = src.to(DEVICE)
        label = label.to(DEVICE)
        yu = yu.to(DEVICE)
        optimizer.zero_grad()

        intent_output,kl= model(src.long(),yu.float())

        loss3 = loss_intent(intent_output, label)
    # 3.联合slot loss + intent loss
        #print(loss2)
        loss = loss3+kl
        #loss.backward()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        #scaler.step(optimizer)
        #scaler.update()

        epoch_loss += loss.item()
    print("time: ", time.time() - start_time)
    return epoch_loss / len(iterator)


def F(predictions, labels):
    return [f1_score(labels, predictions, average='binary') * 100, f1_score(labels, predictions, average='macro') * 100]

def calculate_metric(gt, pred):

    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc=(((TP+TN)/float(TP+TN+FP+FN)) * 100)
    pre=(((TP)/float(TP+FP)) * 100)
    sn=((TP / float(TP+FN)) * 100)
    sp=((TN / float(TN+FP)) * 100)
    print('ACC:%.3f'% acc)
    print("PRE: %.3f" % pre)
    print('SN:%.3f'% sn)
    print('SP:%.3f'% sp)
    return acc,pre,sn,sp

def predict(model,iterator, loss_intent,count):
    bios_data = os.path.join(base_dir, 'model')
    saved_model_path=os.path.join(bios_data, "modelty3-{0}.pth".format(count))
    model.load_state_dict(torch.load(saved_model_path))
    model=model.to(DEVICE)
    #model=torch.load(saved_model_path).to(DEVICE)
    sum_score,f1,mcc,auc,acc,pre,sn,sp,predicted4,true_intent2=evaluate(model, iterator, loss_intent)
    return sum_score,f1,mcc,auc,acc,pre,sn,sp,predicted4,true_intent2
def evaluate(model, iterator,  loss_intent):
    model.eval()

    predicted = []
    predicted2 = []
    predicted3 = []
    true_label = []
    true_intent = []
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, _ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
            label = batch.target
            src = src.to(DEVICE)
            yu, _ = batch.yufa
            yu = yu.to(DEVICE)
            intent_output,kl= model(src.long(),yu.long())  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]


            _,output2  = torch.max(intent_output, 1)
            #output3,_ = torch.max(intent_output, 1)
            output3=intent_output[:,1]
            output2 = output2.cpu()
            output3 = output3.cpu()
            predicted2.extend(output2)
            true_intent.extend(label)
            predicted3.extend(output3)

        predicted2 = np.array(predicted2, dtype=np.float64)
        true_intent = np.array(true_intent, dtype=np.float64)
        predicted3 = np.array(predicted3, dtype=np.float64)
        predicted4=predicted3
        true_intent2=true_intent
        f1 = F(predicted2, true_intent)[0]
        mcc = matthews_corrcoef(true_intent,predicted2)* 100
        auc = metrics.roc_auc_score(true_intent,predicted3)* 100
        acc,pre,sn,sp=calculate_metric(true_intent,predicted2)
        print("F1: %.3f" % f1)
        #print("PRE: %.3f" % pre)
        #print("SN: %.3f" % sn)
        print("MCC: %.3f" % mcc)
        print("AUC:%.3f" % auc)
        sum_score=(f1+mcc+auc+acc+pre+sn+sp)/7
        print("sum score:%.3f" % sum_score)


    return sum_score,f1,mcc,auc,acc,pre,sn,sp,predicted4,true_intent2


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


n_epochs =50 # 迭代次数
clip = 0.1  # 梯度裁剪
lr_max = 0.00005
lr_min = 0

best_valid_loss = float('inf')
best_score = 0
best_ac = 0
eva_score0=1
best_loss=10
er=1
zhiling=input("是否做验证/下游/训练？1/2/0：")
if zhiling=='1':
    count=input("序号：")
    sum_score,f1,mcc,auc,acc,pre,sn,sp,predicted4,true_intent2=predict(model,val_iter, loss_intent,count)


