import os

from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import f1_score, roc_curve
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
atis_data = os.path.join(base_dir, 'bio')
from sklearn.metrics import roc_auc_score
import torch.nn.functional as Fn
import torch
'''
build train and val dataset
'''
tokenize = lambda s: s.split()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
# def string_to_float(string_list):
#     return [float(string) for string in string_list]

# YUFA = data.Field(sequential=True, 
#                   use_vocab=False,
#                   batch_first=True,
#                   dtype=torch.float,
#                   preprocessing=string_to_float,
#                   include_lengths=True)
YUFA = data.Field(sequential=True, tokenize=tokenize,
                    lower=True, use_vocab=True,
                    init_token='[SOS]',
                    pad_token='<pad>', unk_token='<unk>',
                    batch_first=True, fix_length=555,
                    include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence
SOURCE = data.Field(sequential=True, tokenize=tokenize,
                    lower=True, use_vocab=True,
                    init_token='[SOS]',
                    pad_token='<pad>', unk_token='<unk>',
                    batch_first=True, fix_length= 5000,
                    include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence

LABEL = data.Field(
    sequential=False,unk_token=None,
    use_vocab=True)

train, val, test = data.TabularDataset.splits(
    path=atis_data,
    skip_header=True,
    train='biosyu3.train.csv',
    validation='biosyu3.valid.csv',
    test='biosyu3.test.csv',
    format='csv',
    fields=[('source', SOURCE), ('target', LABEL), ('yufa', YUFA)])

SOURCE.build_vocab(train, val, test)
LABEL.build_vocab(train, val, test)
YUFA.build_vocab(train, val, test)

train_iter, val_iter,test_iter = data.Iterator.splits(
    (train, val,test),
    batch_sizes=(92,92,92),  # 训练集设置为32,验证集整个集合用于测试
    shuffle=True,
    sort_within_batch=True,  # 为true则一个batch内的数据会按sort_key规则降序排序
    sort_key=lambda x: len(x.source))  # 这里按src的长度降序排序，主要是为后面pack,pad操作)

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math
import time
import torch.nn.functional as Fn
import torch
import gensim
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
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)
    def forward(self, input):
        return self.maxpool1(input)
class pooling2(nn.Module):
    def __init__(self):
        super(pooling2, self).__init__()
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, ceil_mode=False)
    def forward(self, input):
        return self.avgpool2(input)
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length=5000):
        super(Encoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cache_dir = './'
        cache_dir2 = './'
        

        # 加载预训练的Word2Vec模型
        
        # 获取词向量矩阵和词汇表大小
        #weights = torch.FloatTensor(model.wv.vectors)
        #vocab_size, embedding_dim = weights.size()

        # 初始化nn.Embedding层
        #embedding = nn.Embedding(vocab_size, embedding_dim)

        # 将预训练的词向量赋值给nn.Embedding层
        #embedding.weight.data.copy_(weights)
        glove = GloVe(name='6B', dim=100, cache=cache_dir)
        charngram = CharNGram(cache=cache_dir2)
        self.glove_emb = nn.Embedding.from_pretrained(glove, freeze=True)
        self.charngram = nn.Embedding.from_pretrained(charngram.vectors, freeze=True)
        
        self.tok_embedding = self.glove_emb
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        # 多层encoder
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)
        
        self.w_q = nn.Linear(200, hid_dim).to(device)
        self.w_q2 = nn.Linear(85, hid_dim).to(device)
        self.pool = pooling()
        self.pool2 = pooling2()

        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

    def forward(self, src, src_mask,yu,tag):
        # src:[batch_size, src_len]
        # src_mask:[batch_size, 1, 1, src_len]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x1 = self.glove_emb(src).to(device)
        x2 = self.charngram(src).to(device)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        emb = torch.cat((x1, x2), dim=2).to(device)
        x = self.w_q(emb)
        # 位置信息
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # token编码+位置编码
        src = self.dropout((x*self.scale) + self.pos_embedding(pos))
        src=src.view(src.size(0),1,src.size(1),src.size(2)).to(device)
        src=self.pool2(src).to(device)
        #src=self.pool2(src).to(device)
        src=self.pool(src).to(device)
        src=src.view(src.size(0),src.size(2),src.size(3)).to(device)
        src=self.w_q2(src).to(device)
        i = 0
        kl=0
        for layer in self.layers:
            if i == 0:
                
                src, biaW = layer(src, src_mask)
            else:
                src, biaW2 = layer(src, src_mask)  # [batch_size, src_len, hid_dim]
                if tag!=0:
                    kl = Syntacticdependency(yu, biaW2)
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
yufa_length=277

def Syntacticdependency(in_data,biaffineW):
    #start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_data=in_data.cpu()
    in_data=np.array(in_data,dtype=np.float32)
    pe3=cadoo(in_data)
    pe3 = torch.tensor(pe3).to(device)
    pe3=pe3.view(-1,pe3.size(-1),pe3.size(-1))
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
        
#         Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch_size, query_len, n_heads, head_dim]
#         K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # [batch_size, n_heads, query_len, key_len]
        
#         if mask is not None:
#             energy = energy.mask_fill(mask == 0, -1e10)
        
#         attention = torch.softmax(energy, dim=-1) # [batch_size, n_heads, query_len, key_len]
        
#         x = torch.matmul(self.dropout(attention), V) # [batch_size, n_heads, query_len, head_dim]
        
#         x = x.permute(0, 2, 1, 3).contiguous() # [batch_size, query_len, n_heads, head_dim]
        
#         x = x.view(batch_size, -1, self.hid_dim) # [batch_size, query_len, hid_dim]
        
#         x = self.fc_o(x) # [batch_size, query_len, hid_dim]
        
#         return x,energy
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

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)  # [batch_size, query_len, n_heads, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        cun=Q.size(1)-1
        Q1 = Q[:, cun:, :, :]
        K1 = K[:, cun:, :, :]
        V1 = V[:, cun:, :, :]
        Q = Q[:, 0:cun, :, :]
        K = K[:, 0:cun, :, :]
        V = V[:, 0:cun, :, :]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]
        if mask is not None:
            energy = energy.mask_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, query_len, key_len]
        x = torch.matmul(self.dropout(attention), V)  # [batch_size, n_heads, query_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight = nn.Parameter(torch.randn(batch_size, Q1.size(1), Q1.size(3), Q1.size(3))).to(device)
        Q1 = Q1 @ weight
        energy1 = torch.matmul(Q1, K1.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]
        if mask is not None:
            energy1 = energy1.mask_fill(mask == 0, -1e10)
        attention1 = torch.softmax(energy1, dim=-1)  # [batch_size, n_heads, query_len, key_len]
        x1 = torch.matmul(self.dropout(attention1), V1)  # [batch_size, n_heads, query_len, head_dim]
        x1 = x1.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        x = torch.cat([x, x1], dim=2)
        x = x.view(batch_size, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]
        x = self.fc_o(x)  # [batch_size, query_len, hid_dim]

        return x, energy1

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
        self.encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout)
        self.gelu = nn.GELU()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Sequential(nn.Linear(hid_dim, hid_dim),nn.Dropout(dropout), nn.Tanh(),nn.LayerNorm(hid_dim))
        self.intent_out = nn.Linear(hid_dim, intent_size)
        self.linear = nn.Linear(hid_dim, hid_dim)
        #self.f3 = nn.Linear(hid_dim, hid_dim2)
        #self.mlp= Model(hid_dim,hid_dim, slot_size).to(device)
        #self.LayerN=nn.LayerNorm(hid_dim)
    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]

    def forward(self, src,yu,tag):
        src_mask = self.make_src_mask(src)
        encoder_out,kl = self.encoder(src, src_mask,yu,tag)  # [batch_size, src_len, hid_dim]
        cls_hidden = self.fc(encoder_out[:, 0])  # [batch_size, hid_dim]
        intent_output = self.intent_out(cls_hidden)  # [batch_size, intent_size]
        intent_output=Fn.softmax(intent_output,dim=-1)
        return intent_output,kl,encoder_out

n_layers = 2  # transformer-encoder层数
n_heads = 4 # 多头self-attention
hid_dim =768
dropout = 0.2
pf_dim = 768* 4

input_dim = len(SOURCE.vocab)
intent_size = len(LABEL.vocab)
print(input_dim)
print(intent_size)# intent size
src_pad_idx = SOURCE.vocab.stoi[SOURCE.pad_token]
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
model = BERT(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, intent_size,input_dim, src_pad_idx).to(DEVICE)
model2 = BERT(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, intent_size,input_dim, src_pad_idx).to(DEVICE)
print(f'The model has {count_parameters(model):,} trainable parameters')
# 优化函数
optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01)
# 损失函数(slot)
loss_slot = nn.CrossEntropyLoss()
# ignore_index=src_pad_idx
# 定义损失函数(意图识别)
loss_intent = nn.CrossEntropyLoss()
#model,model2, optimizer = amp.initialize(model,model2, optimizer, opt_level="O0")

def train(model, iterator, optimizer,  loss_intent, clip):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src, _ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
        label = batch.target
        yu, _ = batch.yufa
        src = src.to(DEVICE)
        label = label.to(DEVICE)
        
        # tensor1 = torch.full((yu.size(0), 1), 2)
        # tensor2 = torch.full((yu.size(0), int(5000/27)-44), 1)
        # yu=torch.cat((tensor1, yu, tensor2), dim=1)
        yu = yu.to(DEVICE)
        optimizer.zero_grad()
        intent_output,kl,encoder_out= model(src.long(),yu.float())
        loss3 = loss_intent(intent_output, label)
        loss = 5*kl+loss3
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
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

def predict(iterator, loss_intent,count,model):
    saved_model_path=os.path.join(os.getcwd(), "modelty3f-{0}.pth".format(count))
    model.load_state_dict(torch.load(saved_model_path))
    #model=torch.load(saved_model_path).to(DEVICE)
    model=model.to(DEVICE)
    evaluate(model, iterator, loss_intent)
def predictroc(iterator, loss_intent,count,count2,model,model2):
    saved_model_path1=os.path.join(os.getcwd(), "modelty3f-{0}.pth".format(count))
    saved_model_path2 = os.path.join(os.getcwd(), "modelty3f-{0}.pth".format(count2))
    model.load_state_dict(torch.load(saved_model_path1))
    model2.load_state_dict(torch.load(saved_model_path2))
    #model=torch.load(saved_model_path).to(DEVICE)
    model=model.to(DEVICE)
    model2 = model2.to(DEVICE)
    evaluate(model,model2, iterator, loss_intent)

def evaluate(model,model2, iterator,  loss_intent):
    model.eval()
    
    predicted1 = []
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
            tag=0

            intent_output,kl,encoder_out= model2(src.long(),yu.float(),tag=0)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
            #intent_output = Fn.softmax(intent_output, dim=-1)
            #_,output2  = torch.max(intent_output, 1)
            output3=intent_output[:,1]
            output3 = output3.cpu()
            true_intent.extend(label)
            predicted2.extend(output3)

            intent_output, kl, encoder_out = model(src.long(),
                                                   yu.float(),tag=1)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
            # intent_output = Fn.softmax(intent_output, dim=-1)
            #_, output2 = torch.max(intent_output, 1)
            output3 = intent_output[:, 1]
            output3 = output3.cpu()
            predicted1.extend(output3)

            intent_output, kl, encoder_out = model(yu.long(),
                                                   yu.float(),tag=0)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
            # intent_output = Fn.softmax(intent_output, dim=-1)
            #_, output2 = torch.max(intent_output, 1)
            output3 = intent_output[:, 1]
            output3 = output3.cpu()
            predicted3.extend(output3)


        true_intent = np.array(true_intent, dtype=np.float64)
        predicted1 = np.array(predicted1, dtype=np.float64)
        predicted2 = np.array(predicted2, dtype=np.float64)
        predicted3 = np.array(predicted3, dtype=np.float64)

        y_test, y_pred_1=true_intent, predicted1
        y_test, y_pred_2 = true_intent, predicted2
        y_test, y_pred_3 = true_intent, predicted3

        #auc = metrics.roc_auc_score(true_intent,predicted3)* 100
        fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_1)
        auc1 = roc_auc_score(y_test, y_pred_1)
        fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_2)
        auc2 = roc_auc_score(y_test, y_pred_2)
        fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_3)
        auc3 = roc_auc_score(y_test, y_pred_3)

        plt.figure(figsize=(8,5))
        lw = 2
        plt.plot(fpr1, tpr1, color='darkorange', markersize=4,
                 lw=lw, label='sequence embedding+biological feature (AUC = %0.4f)' % auc1)
        plt.plot(fpr2, tpr2, color='darkgreen', markersize=4,
                 lw=lw, label='sequence embedding                               (AUC = %0.4f)' % auc2)
        plt.plot(fpr3, tpr3, color='darkblue', markersize=4,
                 lw=lw, label='biological feature                                      (AUC = %0.4f)' % auc3)
                 
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1])
        plt.ylim([0, 1.02])
        plt.xticks(np.arange(0, 1, 0.2))
        plt.yticks(np.arange(0, 1.02, 0.2))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")


        #acc,pre,sn,sp=calculate_metric(true_intent,predicted2)
        #print("F1: %.3f" % f1)
        #print("MCC: %.3f" % mcc)
        #print("AUC:%.3f" % auc)
        #sum_score=(f1+mcc+auc+acc+pre+sn+sp)/7
        #print("sum score:%.3f" % sum_score)
        sum_score,f1,mcc,auc,acc,pre,sn,sp=0
        

    return sum_score,f1,mcc,auc,acc,pre,sn,sp

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

n_epochs =100 # 迭代次数
clip = 0.1  # 梯度裁剪
lr_max = 0.0004
lr_min = 0

best_valid_loss = float('inf')
best_score =0
best_ac = 0
eva_score0=1
er=1
zhiling=input("是否做验证/下游/训练？1/2/0：")
if zhiling=='1':
    count=input("序号：")
    count2 = input("序号：")
    #predict(train_iter, loss_intent,count,model)
    #predict(val_iter, loss_intent,count,model)
    predictroc(test_iter, loss_intent,count,count2,model,model2)
elif zhiling=='2':
    count=input("序号：")
    saved_model_path=os.path.join(os.getcwd(), "modelty3-{0}.pth".format(count))
    model.load_state_dict(torch.load(saved_model_path))
    model=model.to(DEVICE)
    for epoch in range(n_epochs):
        epoch=epoch+1
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=n_epochs, lr_min=lr_min,
                             lr_max=lr_max,
                             warmup=True)
        start_time = time.time()
        loss = train(model, train_iter, optimizer, loss_intent, clip)
        print('sum loss',loss)
        if epoch%5==0:
            eva_score,f1,mcc,auc,acc,pre,sn,sp = evaluate(model, train_iter,  loss_intent)
            eva_score,f1,mcc,auc,acc,pre,sn,sp = evaluate(model, val_iter,  loss_intent)
            eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, test_iter,  loss_intent)
        else:
            eva_score,f1,mcc,auc,acc,pre,sn,sp = evaluate(model, val_iter,  loss_intent)
            eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, test_iter,  loss_intent)
            
        if eva_score0>best_score:
            best_score=eva_score0
        # if loss<best_score:
        #     best_score=loss
            model_path = os.path.join(os.getcwd(), "model6w-{0}.pth".format(epoch))
            model_path2 = os.path.join(os.getcwd(), "model6w-{0}.pth".format(epoch-er))
            torch.save(model.state_dict(), model_path)
            #torch.save(model, model_path)
            if (epoch-1)!=0:
                os.unlink(model_path2)
                er=1
        else:
            er+=1
        end_time = time.time()
        print("time: ", time.time() - start_time)
else:
    for epoch in range(n_epochs):
            epoch=epoch+1
            adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=n_epochs, lr_min=lr_min,
                                 lr_max=lr_max,
                                 warmup=True)
            start_time = time.time()
            loss = train(model, train_iter, optimizer, loss_intent, clip)
            print("sum loss",loss)
            if epoch%5==0:
                eva_score,f1,mcc,auc,acc,pre,sn,sp = evaluate(model, train_iter,  loss_intent)
                eva_score,f1,mcc,auc,acc,pre,sn,sp = evaluate(model, val_iter,  loss_intent)
                eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, test_iter,  loss_intent)
            else:
                eva_score,f1,mcc,auc,acc,pre,sn,sp = evaluate(model, val_iter,  loss_intent)
                eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, test_iter,  loss_intent)
            if eva_score0>best_score:
                best_score=eva_score0
                model_path = os.path.join(os.getcwd(), "modelty5-{0}.pth".format(epoch))
                model_path2 = os.path.join(os.getcwd(), "modelty5-{0}.pth".format(epoch-er))
                torch.save(model.state_dict(), model_path)
                if (epoch-1)!=0:
                    os.unlink(model_path2)
                    er=1
            else:
                er+=1
            end_time = time.time()
            print("time: ", time.time() - start_time)