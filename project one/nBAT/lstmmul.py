import torch
import torch.nn as nn
import torch.optim as optim
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
from mlp import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import torch.nn.functional as Fn


base_dir = os.path.abspath(os.path.join(os.getcwd()))
atis_data = os.path.join(base_dir, 'data')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    def __init__(self,input_dim, hid_dim, dropout,src_pad_idx,max_length=2200):
        super(Encoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.src_pad_idx= src_pad_idx
        cache_dir = './'
        cache_dir2 = './'


        self.word_embeddings = nn.Embedding(input_dim,hid_dim, padding_idx=self.src_pad_idx)
        self.word_embeddings.weight.data.uniform_(-1., 1.)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)
        #self.w_q = nn.Linear(200, hid_dim).to(device)
        #self.w_q2 = nn.Linear(11, hid_dim).to(device)
        # 多层encoder
        self.pool = pooling()
        self.pool2 = pooling2()
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

    def forward(self, src):
        # src:[batch_size, src_len]
        # src_mask:[batch_size, 1, 1, src_len]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = self.word_embeddings(src).to(device)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        #x = self.w_q(emb)
        # 位置信息
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # token编码+位置编码
        src = self.dropout(x*self.scale+self.pos_embedding(pos))
        #src = self.dropout(x)
        src=src.transpose(1,2).to(device)
        src=self.pool2(src).to(device)
        src=self.pool(src).to(device)
        src=src.transpose(1,2).to(device)
        # src =src.reshape(src.size(0),200*55,-1)
        # src =self.pool2(src).to(device)
        # src =src.reshape(int(src.size(0)/3),200*55,-1)
        # src =self.pool(src).to(device)
        # src =src.squeeze() 
        # src =src.reshape(src.size(0),55,-1)
        return src

class AttentionLSTM(nn.Module):
    def __init__(self,src_pad_idx,vocab_size, input_size, hidden_size, num_layers, num_classes,dropout):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, self.hidden_size// 2, num_layers=self.num_layers, bidirectional=True,batch_first=True,dropout=self.dropout)
        self.f3 = Encoder(vocab_size,input_size,dropout,src_pad_idx)
        self.LinearLayer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.Linear( self.hidden_size*2,  self.hidden_size),
            nn.Linear( self.hidden_size, num_classes)
        )
    def attention(self, lstm_out, final_state):
        hidden = final_state.view(-1, self.hidden_size, 1)
        attn_weights = torch.bmm(lstm_out, hidden).squeeze(2)
        soft_attn_weights = Fn.softmax(attn_weights, 1)
        context = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2) 
        return context 
    def forward(self, x,yu):
        x=self.f3(x).to(DEVICE)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size// 2).to(DEVICE)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size// 2).to(DEVICE)
        x=x.to(DEVICE)
        h0 = h0.to(DEVICE)  # 2 for bidirection
        c0 = c0.to(DEVICE)
        out,(h1, c1) = self.lstm(x, (h0, c0))
        out = self.attention(out,h1)
        out = self.LinearLayer(out).to(DEVICE)     

        return out


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
                  pad_token=None, unk_token=None,
                  batch_first=True, fix_length=None,
                  include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence

LABEL = data.Field(
    sequential=False, unk_token=None,
    use_vocab=True)

train, val, test = data.TabularDataset.splits(
    path=atis_data,
    skip_header=True,
    train='biosyu3L.train.csv',
    validation='biosyu3L.valid.csv',
    test='biosyu3L.test.csv',
    format='csv',
    fields=[('source', SOURCE), ('target', LABEL), ('yufa', YUFA)])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOURCE.build_vocab(train, val, test)
LABEL.build_vocab(train, val, test)
YUFA.build_vocab(train, val, test)

train_iter, val_iter,test_iter = data.Iterator.splits(
    (train, val,test),
    batch_sizes=(128,128,128),  # 训练集设置为32,验证集整个集合用于测试
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
input_size = 200
hidden_size = 768
num_layers = 1
num_classes = 2
dropout=0
vocab_size = len(SOURCE.vocab)
intent_size = len(LABEL.vocab)
print(vocab_size)
print(intent_size)  # intent size
src_pad_idx = SOURCE.vocab.stoi[SOURCE.pad_token]
model = AttentionLSTM(src_pad_idx,vocab_size,input_size,hidden_size,num_layers,num_classes,dropout).to(DEVICE)
model3=model
#
#
# # lv = LABEL.vocab.itos[0]
# # print(lv)
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# print(f'The model has {count_parameters(model):,} trainable parameters')
# # 优化函数
# optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01)
# # 损失函数(slot)
# loss_slot = nn.CrossEntropyLoss()
# # ignore_index=src_pad_idx
# # 定义损失函数(意图识别)
# loss_intent = nn.CrossEntropyLoss()
# #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#
#
# def train(model, iterator, optimizer, loss_intent, clip):
#     start_time = time.time()
#     model.train()
#     epoch_loss = 0
#
#     # scaler = GradScaler()
#     for i, batch in enumerate(iterator):
#         src, _ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
#         label = batch.target
#         yu, _ = batch.yufa
#         src = src.to(DEVICE)
#         label = label.to(DEVICE)
#         yu = yu.to(DEVICE)
#         optimizer.zero_grad()
# #         src = src.squeeze()
#
# #         if src.size(0) < 384:
# #             continue
# #         label = label[0::3][:128]
#
#         intent_output = model(src.long(),yu).to(DEVICE)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
#         loss2 = loss_intent(intent_output, label)
#         loss = loss2
#         #print(loss)
#         loss.backward()
#         # with amp.scale_loss(loss, optimizer) as scaled_loss:
#         #     scaled_loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         # scaler.step(optimizer)
#         # scaler.update()
#         epoch_loss += loss.item()
#     print("time: ", time.time() - start_time)
#     return epoch_loss / len(iterator)
#
#
# def F(predictions, labels):
#     return [f1_score(labels, predictions, average='binary') * 100, f1_score(labels, predictions, average='macro') * 100]
#
#
# def calculate_metric(gt, pred):
#     confusion = confusion_matrix(gt, pred)
#     TP = confusion[1, 1]
#     TN = confusion[0, 0]
#     FP = confusion[0, 1]
#     FN = confusion[1, 0]
#     acc = (((TP + TN) / float(TP + TN + FP + FN)) * 100)
#     pre = (((TP) / float(TP + FP)) * 100)
#     sn = ((TP / float(TP + FN)) * 100)
#     sp = ((TN / float(TN + FP)) * 100)
#     print('ACC:%.3f' % acc)
#     print("PRE: %.3f" % pre)
#     print('SN:%.3f' % sn)
#     print('SP:%.3f' % sp)
#     return acc, pre, sn, sp
#
#
#
#
# def evaluate(model, iterator, loss_intent):
#     model.eval()
#
#     predicted = []
#     predicted2 = []
#     predicted3 = []
#     true_label = []
#     true_intent = []
#     epoch_loss = 0
#
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             src, _ = batch.source  # src=[batch_size, seq_len]，这里batch.src返回src和src的长度，因为在使用torchtext.Field时设置include_lengths=True
#             label = batch.target
#             src = src.to(DEVICE)
#             yu, _ = batch.yufa
#             yu = yu.to(DEVICE)
# #             src = src.squeeze()
#
# #             if src.size(0) < 384:
# #                 continue
# #             label = label[0::3][:128]
#             intent_output = model(src.long(),yu).to(DEVICE)  # [batch_size, intent_dim]; [batch_size, trg_len-1, slot_size]
#             _, output2 = torch.max(intent_output, 1)
#             # output3,_ = torch.max(intent_output, 1)
#             output3 = intent_output[:, 1]
#             output2 = output2.cpu()
#             output3 = output3.cpu()
#             predicted2.extend(output2)
#             true_intent.extend(label)
#             predicted3.extend(output3)
#
#         predicted2 = np.array(predicted2, dtype=np.float64)
#         true_intent = np.array(true_intent, dtype=np.float64)
#         predicted3 = np.array(predicted3, dtype=np.float64)
#
#         f1 = F(predicted2, true_intent)[0]
#         mcc = matthews_corrcoef(true_intent, predicted2) * 100
#         auc = metrics.roc_auc_score(true_intent, predicted3) * 100
#         acc, pre, sn, sp = calculate_metric(true_intent, predicted2)
#         print("F1: %.3f" % f1)
#         # print("PRE: %.3f" % pre)
#         # print("SN: %.3f" % sn)
#         print("MCC: %.3f" % mcc)
#         print("AUC:%.3f" % auc)
#         sum_score = (f1 + mcc + auc + acc + pre + sn + sp) / 7
#         print("sum score:%.3f" % sum_score)
#
#     return sum_score, f1, mcc, auc, acc, pre, sn, sp
#
#
# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
#
#
# n_epochs = 50# 迭代次数
# clip = 0.1  # 梯度裁剪
# lr_max = 0.00005
# lr_min = 0
#
# best_valid_loss = float('inf')
# best_score = 0
# best_ac = 0
# er = 1
# zhiling = input("是否做验证/下游/训练？1/2/0：")
# def predict(model,iterator, loss_intent, count):
#     bios_data = os.path.join(base_dir, 'model')
#     saved_model_path = os.path.join(bios_data, "modelt3-{0}.pth".format(count))
#     model.load_state_dict(torch.load(saved_model_path))
#     #model = torch.load(saved_model_path).to(DEVICE)
#     evaluate(model, iterator, loss_intent)
# if zhiling == '1':
#     count = input("序号：")
#     predict(model,train_iter, loss_intent, count)
#     predict(model,val_iter, loss_intent, count)
#     predict(model,test_iter, loss_intent, count)


