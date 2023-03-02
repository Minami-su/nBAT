import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
import os
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import f1_score
from torchtext import data, datasets
from sklearn import metrics
from torchtext.vocab import GloVe, CharNGram
from numba import jit
from apex import amp
from Warmup import adjust_learning_rate

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import torch.nn.functional as Fn
from sklearn.metrics import matthews_corrcoef
from torch.autograd import Variable
base_dir = os.path.abspath(os.path.join(os.getcwd()))
atis_data = os.path.join(base_dir, 'data')
import random
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
build train and val dataset
'''

tokenize = lambda s: s.split()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
SOURCE = data.Field(sequential=True, tokenize=tokenize,
                    lower=True, use_vocab=True,
                    init_token=None,
                    pad_token='<pad>', unk_token=None,
                    batch_first=True, fix_length=2160,
                    include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence
YUFA = data.Field(sequential=True, tokenize=tokenize,
                  lower=True, use_vocab=True,
                  init_token=None,
                  pad_token='<pad>', unk_token=None,
                  batch_first=True, fix_length=240,
                  include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence


train, val, test = data.TabularDataset.splits(
    path=atis_data,
    skip_header=True,
    train='case2.csv',
    validation='case2.csv',
    test='case2.csv',
    format='csv',
    fields=[('source', SOURCE), ('yufa', YUFA)])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOURCE.build_vocab(train, val, test)
YUFA.build_vocab(train, val, test)

train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test),
    batch_sizes=(3, 3, 3),  # 训练集设置为32,验证集整个集合用于测试
    shuffle=False,
    sort_within_batch=True,  # 为true则一个batch内的数据会按sort_key规则降序排序
    sort_key=lambda x: len(x.source))  # 这里按src的长度降序排序，主要是为后面pack,pad操作)

class pooling(nn.Module):
    def __init__(self):
        super(pooling, self).__init__()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        return self.maxpool1(input)


class pooling2(nn.Module):
    def __init__(self):
        super(pooling2, self).__init__()
        self.avgpool2 = nn.AvgPool1d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        return self.avgpool2(input)


class Encoder(nn.Module):
    def __init__(self, hid_dim, dropout, src_pad_idx,vocab_size,max_length=2200):
        super(Encoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.src_pad_idx = src_pad_idx
        cache_dir = './'
        cache_dir2 = './'
        self.word_embeddings = nn.Embedding(vocab_size,hid_dim, padding_idx=self.src_pad_idx)
        self.word_embeddings.weight.data.uniform_(-1., 1.)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

        # 多层encoder
        self.pool = pooling()
        self.pool2 = pooling2()
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        return src_mask

    def forward(self, src):
        # src:[batch_size, src_len]
        # src_mask:[batch_size, 1, 1, src_len]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src2 = src.clone()
        src2 = src2.float()
        # src2 =src2.reshape(src2.size(0),55,-1)
        # src2 =self.avg_pool(src2).to(device)
        # src2 =src2.reshape(int(src2.size(0)/3),55,-1)
        # src2 =self.max_pool(src2).to(device)
        src2 = self.pool2(src2).to(device)
        src2 = self.pool(src2).to(device)
        src_mask = self.make_src_mask(src2)

        x = self.word_embeddings(src).to(device)
        #x2 = self.charngram(src).to(device)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        #emb = torch.cat((x1, x2), dim=2).to(device)
        # 位置信息
        #x = emb
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # token编码+位置编码
        src = self.dropout((x * self.scale) + self.pos_embedding(pos))
        src = src.transpose(1, 2).to(device)
        # src =src.reshape(src.size(0),200*55,-1)
        # src =self.pool2(src).to(device)
        # src =src.reshape(int(src.size(0)/3),200*55,-1)
        # src =self.pool(src).to(device)
        # src =src.squeeze() 
        # src =src.reshape(src.size(0),55,-1)
        src = self.pool2(src).to(device)
        src = self.pool(src).to(device)
        src = src.transpose(1, 2).to(device)

        return src, src_mask


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

def featureanalysis(in_data,biaffineW):
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

    # t1 = t1[:, 1:, 1:]
    # pe3 = pe3[:, 1:, 1:]

    t1 = Fn.log_softmax(t1, dim=-1)
    

    kl = (1 / T) * Fn.kl_div(t1.float(),pe3.float(), reduction='batchmean')

    
    return kl

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
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, query_len, key_len]
        x = torch.matmul(self.dropout(attention), V)  # [batch_size, n_heads, query_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight = nn.Parameter(torch.randn(batch_size, Q1.size(1), Q1.size(3), Q1.size(3))).to(device)
        Q1 = Q1 @ weight
        energy1 = torch.matmul(Q1, K1.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]
        if mask is not None:
            energy1 = energy1.masked_fill(mask == 0, -1e10)
        attention1 = torch.softmax(energy1, dim=-1)  # [batch_size, n_heads, query_len, key_len]
        x1 = torch.matmul(self.dropout(attention1), V1)  # [batch_size, n_heads, query_len, head_dim]
        x1 = x1.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        x = torch.cat([x, x1], dim=2)
        x = x.view(batch_size, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]
        x = self.fc_o(x)  # [batch_size, query_len, hid_dim]

        return x, energy1

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
#
#     def forward(self, query, key, value, mask=None):
#         batch_size = query.shape[0]
#         # query:[batch_size, query_len, hid_dim]
#         # key:[batch_size, query_len, hid_dim]
#         # value:[batch_size, query_len, hid_dim]
#         Q = self.fc_q(query)
#         K = self.fc_k(key)
#         V = self.fc_v(value)
#
#         Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
#                                                                         3)  # [batch_size, query_len, n_heads, head_dim]
#         K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]
#
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, -1e10)
#
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, query_len, key_len]
#
#         x = torch.matmul(self.dropout(attention), V)  # [batch_size, n_heads, query_len, head_dim]
#
#         x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
#
#         x = x.view(batch_size, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]
#
#         x = self.fc_o(x)  # [batch_size, query_len, hid_dim]
#
#         return x, energy


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



class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size,input_size, hidden_size, num_layers, num_classes,
                 n_layers,
                 n_heads,  # 多头self-attention
                 hid_dim,
                 dropout,
                 pf_dim, 
                 src_pad_idx):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True,batch_first=True,dropout=self.dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc7 = nn.Linear(hidden_size * 2, hidden_size)
        
        self.f3 = Encoder(input_size, dropout, src_pad_idx,vocab_size)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.fc6 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Dropout(dropout), nn.Tanh(), nn.LayerNorm(hid_dim))
        self.fc8 = nn.Linear(hidden_size,input_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size*4),
            nn.Linear( self.hidden_size*4,  self.hidden_size*2),
            nn.Linear( self.hidden_size*2, num_classes)
        )
    def attention(self, out, h1):
        h1 = h1.view(-1, self.hidden_size*2, 1)
        attn_weight = torch.bmm(out, h1).squeeze(2)
        soft_attn_weight = Fn.softmax(attn_weight, dim=1)
        out = torch.bmm(out.transpose(1, 2), soft_attn_weight.unsqueeze(2)).squeeze(2) 
        return out   
    def forward(self, x,yu):

        x, src_mask = self.f3(x)
        #print(x.shape)
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).to(DEVICE)  # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).to(DEVICE)
        x = x.to(DEVICE)
        h0 = h0.to(DEVICE)  # 2 for bidirection
        c0 = c0.to(DEVICE)
        out,_= self.lstm(x, (h0, c0))
        out = torch.tanh(self.attn(out))
        
        #out = torch.tanh(self.attention(out,h1))
        src = self.fc7(out)
        i = 0
        kl = 0
        for layer in self.layers:
            if i == 0:
                src, biaW = layer(src, src_mask)
            else:
                src, biaW2 = layer(src, src_mask)  # [batch_size, src_len, hid_dim]
                kl = featureanalysis(yu, biaW2)
            i += 1
        out = self.fc6(src)
        out = self.fc8(out)
        out,(h1, c1) = self.lstm(out, (h0, c0))
        #out = torch.tanh(self.attn(out))
        out = self.attention(out,h1)
        #out = self.fc(out[:, -1, :]).to(DEVICE)
        out = self.mlp(out).to(DEVICE)
        out = Fn.softmax(out, dim=-1)
        return out, kl






vocab_size = len(SOURCE.vocab)

print(vocab_size)

src_pad_idx = SOURCE.vocab.stoi[SOURCE.pad_token]
# lv = LABEL.vocab.itos[0]
# print(lv)
print(src_pad_idx)

input_size = 200
hidden_size = 768
layer =1
num_class = 2

n_layers = 2  # transformer-encoder层数
n_heads = 4  # 多头self-attention
hid_dim = 768
dropout = 0.2
pf_dim = 768 * 4
model = AttentionLSTM(vocab_size,input_size, hidden_size, layer, num_class,
                      n_layers,
                      n_heads,  # 多头self-attention
                      hid_dim,
                      dropout,
                      pf_dim,
                      src_pad_idx).to(DEVICE)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# 优化函数
optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01)
# 损失函数(slot)
loss_slot = nn.CrossEntropyLoss()
# ignore_index=src_pad_idx
# 定义损失函数(意图识别)
loss_intent = nn.CrossEntropyLoss()


# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


def train(model, iterator, optimizer, loss_intent, clip):
    start_time = time.time()
    model.train()
    epoch_loss = 0

    # scaler = GradScaler()
    for i, batch in enumerate(iterator):
        src,_ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
        label = batch.target
        yu,_ = batch.yufa
        src = src.to(DEVICE)
        label = label.to(DEVICE)
        yu = yu.to(DEVICE)
        optimizer.zero_grad()
        #src = src.squeeze()
        
        # if src.size(0) < 384:
        #     continue
        # label=label[:128]
        #src = autobatchsize(src)
        intent_output, kl = model(src.long(),yu)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
        
        # print(intent_output.shape)
        loss2 = loss_intent(intent_output, label)
        loss = loss2+0.5*kl
        # print(loss)
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        epoch_loss += loss.item()
    print("time: ", time.time() - start_time)
    return epoch_loss / len(iterator)


def F(predictions, labels):
    return [f1_score(labels, predictions, average='binary') * 100, f1_score(labels, predictions, average='macro') * 100]


def calculate_metric(gt, pred):
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc = (((TP + TN) / float(TP + TN + FP + FN)) * 100)
    pre = (((TP) / float(TP + FP)) * 100)
    sn = ((TP / float(TP + FN)) * 100)
    sp = ((TN / float(TN + FP)) * 100)
    print('ACC:%.3f' % acc)
    print("PRE: %.3f" % pre)
    print('SN:%.3f' % sn)
    print('SP:%.3f' % sp)
    return acc, pre, sn, sp





def evaluate(model, iterator, loss_intent):
    model.eval()

    predicted = []
    predicted2 = []
    predicted3 = []
    true_label = []
    true_intent = []
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src,_ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
            src = src.to(DEVICE)
            yu,_= batch.yufa
            yu = yu.to(DEVICE)
            src = src.to(DEVICE)
            # if src.size(0) < 384:
            #     continue
            # label=label[:128]
            intent_output, kl = model(src.long(),yu)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]

            _, output2 = torch.max(intent_output, 1)
            #print(intent_output.shape)
            output3,_ = torch.max(intent_output, 1)
            sum=0
            #print(intent_output)
            list=[]
            for i in range(3):
                for j in range(2160):
                    x=src[i][j]
                    if x==0:
                        break
                    list.append(SOURCE.vocab.itos[x])
                list.append('sep')
            #print(list)
            for i in range(3):
                sum+=float(intent_output[i][1])
                #print(intent_output[i][0])
            print(sum/3.0)
        #     # output3,_ = torch.max(intent_output, 1)
        #     output3 = intent_output[:, 1]
        #     output2 = output2.cpu()
        #     output3 = output3.cpu()
        #     predicted2.extend(output2)
        #     true_intent.extend(label)
        #     predicted3.extend(output3)
        #
        # predicted2 = np.array(predicted2, dtype=np.float64)
        # true_intent = np.array(true_intent, dtype=np.float64)
        # predicted3 = np.array(predicted3, dtype=np.float64)
        #
        # f1 = F(predicted2, true_intent)[0]
        # mcc = matthews_corrcoef(true_intent, predicted2) * 100
        # auc = metrics.roc_auc_score(true_intent, predicted3) * 100
        # acc, pre, sn, sp = calculate_metric(true_intent, predicted2)
        # print("F1: %.3f" % f1)
        # # print("PRE: %.3f" % pre)
        # # print("SN: %.3f" % sn)
        # print("MCC: %.3f" % mcc)
        # print("AUC:%.3f" % auc)
        # sum_score = (f1 + mcc + auc + acc + pre + sn + sp) / 7
        # print("sum score:%.3f" % sum_score)
        sum_score, f1, mcc, auc, acc, pre, sn, sp=0,0,0,0,0,0,0,0

    return sum_score, f1, mcc, auc, acc, pre, sn, sp

def evaluate2(model, iterator, loss_intent):
    model.train()

    predicted = []
    predicted2 = []
    predicted3 = []
    true_label = []
    true_intent = []
    epoch_loss = 0


    for i, batch in enumerate(iterator):
        src,_ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
        src = src.to(DEVICE)
        yu,_= batch.yufa
        yu = yu.to(DEVICE)
        src = src.to(DEVICE)
        # if src.size(0) < 384:
        #     continue
        # label=label[:128]

        intent_output, kl = model(src.long(),yu)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
        #print(kl)
        loss=kl
        loss.requires_grad_(True)
        loss.backward()
        _, output2 = torch.max(intent_output, 1)
        #print(intent_output.shape)
        output3,_ = torch.max(intent_output, 1)
        sum=0
        #print(intent_output)
        list=[]
        for i in range(3):
            for j in range(2160):
                x=src[i][j]
                if x==0:
                    break
                list.append(SOURCE.vocab.itos[x])
            list.append('sep')
        #print(list)
        for i in range(3):
            sum+=float(intent_output[i][1])
            #print(intent_output[i][0])
        print(sum/3.0)
    #     # output3,_ = torch.max(intent_output, 1)
    #     output3 = intent_output[:, 1]
    #     output2 = output2.cpu()
    #     output3 = output3.cpu()
    #     predicted2.extend(output2)
    #     true_intent.extend(label)
    #     predicted3.extend(output3)
    #
    # predicted2 = np.array(predicted2, dtype=np.float64)
    # true_intent = np.array(true_intent, dtype=np.float64)
    # predicted3 = np.array(predicted3, dtype=np.float64)
    #
    # f1 = F(predicted2, true_intent)[0]
    # mcc = matthews_corrcoef(true_intent, predicted2) * 100
    # auc = metrics.roc_auc_score(true_intent, predicted3) * 100
    # acc, pre, sn, sp = calculate_metric(true_intent, predicted2)
    # print("F1: %.3f" % f1)
    # # print("PRE: %.3f" % pre)
    # # print("SN: %.3f" % sn)
    # print("MCC: %.3f" % mcc)
    # print("AUC:%.3f" % auc)
    # sum_score = (f1 + mcc + auc + acc + pre + sn + sp) / 7
    # print("sum score:%.3f" % sum_score)
    sum_score, f1, mcc, auc, acc, pre, sn, sp=0,0,0,0,0,0,0,0

    return sum_score, f1, mcc, auc, acc, pre, sn, sp
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


n_epochs = 50 # 迭代次数
clip = 0.1  # 梯度裁剪
lr_max = 0.00005
lr_min = 0

best_valid_loss = float('inf')
best_score = 0
best_ac = 0
er = 1

def predict(model,iterator, loss_intent, count):
    bios_data = os.path.join(base_dir, 'model')
    saved_model_path = os.path.join(bios_data, "modelt4-{0}.pth".format(count))

    model.load_state_dict(torch.load(saved_model_path))

    #model = torch.load(saved_model_path).to(DEVICE)
    evaluate(model, iterator, loss_intent)

def predict2(model, iterator, loss_intent, count):
    bios_data = os.path.join(base_dir, 'model')
    saved_model_path = os.path.join(bios_data, "modelt4-{0}.pth".format(count))

    model.load_state_dict(torch.load(saved_model_path))

    # model = torch.load(saved_model_path).to(DEVICE)
    evaluate2(model, iterator, loss_intent)
zhiling = input("是否做验证/下游/训练？1/2/0：")
if zhiling == '1':
    count = input("序号：")
    print("train score:")
    predict2(model, train_iter, loss_intent, count)
    print("val score:")
    predict(model, val_iter, loss_intent, count)
    print("test score:")
    predict(model, test_iter, loss_intent, count)
elif zhiling == '2':
    count = input("序号：")
    bios_data = os.path.join(base_dir, 'model')
    saved_model_path = os.path.join(bios_data, "modelt4-{0}.pth".format(count))
    model.load_state_dict(torch.load(saved_model_path))
    #model = torch.load(saved_model_path).to(DEVICE)
    for epoch in range(n_epochs):
        epoch = epoch + 1
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=n_epochs, lr_min=lr_min,
                             lr_max=lr_max,
                             warmup=True)
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, loss_intent, clip)
        print(train_loss)
        if epoch%10==0:
            eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, train_iter,  loss_intent)
            eva_score, f1, mcc, auc, acc, pre, sn, sp = evaluate(model, val_iter, loss_intent)
            eva_score0, f10, mcc0, auc0, acc0, pre0, sn0, sp0 = evaluate(model, test_iter, loss_intent)
        else:
            # eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, train_iter,  loss_intent)
            eva_score, f1, mcc, auc, acc, pre, sn, sp = evaluate(model, val_iter, loss_intent)
            eva_score0, f10, mcc0, auc0, acc0, pre0, sn0, sp0 = evaluate(model, test_iter, loss_intent)
        if eva_score0 > best_score:
            best_score = eva_score0
            model_path = os.path.join(bios_data, "modelt4-{0}.pth".format(epoch))
            model_path2 = os.path.join(bios_data, "modelt4-{0}.pth".format(epoch - er))
            torch.save(model.state_dict(), model_path)
            # torch.save(model, model_path)
            if (epoch - 1) != 0:
                os.unlink(model_path2)
                er = 1
        else:
            er += 1
        end_time = time.time()
        print("time: ", time.time() - start_time)

else:
    for epoch in range(n_epochs):
        epoch = epoch + 1
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=n_epochs, lr_min=lr_min,
                             lr_max=lr_max,
                             warmup=True)
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, loss_intent, clip)
        print(train_loss)
        if epoch%10==0:
            eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, train_iter,  loss_intent)
            eva_score, f1, mcc, auc, acc, pre, sn, sp = evaluate(model, val_iter, loss_intent)
            eva_score0, f10, mcc0, auc0, acc0, pre0, sn0, sp0 = evaluate(model, test_iter, loss_intent)
        else:
            # eva_score0,f10,mcc0,auc0,acc0,pre0,sn0,sp0 = evaluate(model, train_iter,  loss_intent)
            eva_score, f1, mcc, auc, acc, pre, sn, sp = evaluate(model, val_iter, loss_intent)
            eva_score0, f10, mcc0, auc0, acc0, pre0, sn0, sp0 = evaluate(model, test_iter, loss_intent)
        if eva_score0 > best_score:
            best_score = eva_score0
            bios_data = os.path.join(base_dir, 'model')
            model_path = os.path.join(bios_data, "modelt4-{0}.pth".format(epoch))
            model_path2 = os.path.join(bios_data, "modelt4-{0}.pth".format(epoch - er))
            torch.save(model.state_dict(), model_path)
            # torch.save(model, model_path)
            if (epoch - 1) != 0:
                os.unlink(model_path2)
                er = 1
        else:
            er += 1
        end_time = time.time()
        print("time: ", time.time() - start_time)

