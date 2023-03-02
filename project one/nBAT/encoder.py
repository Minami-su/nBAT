import torch
from torch import nn
from mlp import Model
import torch.nn.functional as Fn
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
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, src_pad_idx, max_length=2200):
        super(Encoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.src_pad_idx = src_pad_idx
        cache_dir = './'
        cache_dir2 = './'

        self.word_embeddings = nn.Embedding(input_dim, hid_dim, padding_idx=self.src_pad_idx)
        self.word_embeddings.weight.data.uniform_(-1., 1.)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        # self.w_q = nn.Linear(200, hid_dim).to(device)
        # self.w_q2 = nn.Linear(85, hid_dim).to(device)
        # 多层encoder
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.pool = pooling()
        self.pool2 = pooling2()
        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]

    def seq_mask(self,seq):
        batch_size, seq_len = seq.size()
        sub_seq_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
        return sub_seq_mask

    def forward(self, src, src_mask, yu):
        # src:[batch_size, src_len]
        # src_mask:[batch_size, 1, 1, src_len]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src2 = src.clone()
        src2 = src2.float()
        src2 = self.pool2(src2).to(device)
        src2 = self.pool(src2).to(device)
        src_mask = self.make_src_mask(src2)
        print(src_mask)
        # pad_mask = (src == 1).to(DEVICE)
        # y=torch.tensor(-float('inf')).to(DEVICE)
        # pad_mask,y, src=pad_mask,y, src.float()
        # src=torch.where(pad_mask,y, src).to(DEVICE)
        src_mask = None
        # inf_count = torch.sum(torch.eq(src, -float('inf'))).item()
        # print(inf_count)
        # src=src.transpose(0,1).to(DEVICE)
        # print(src)
        # src=src.view(src.size(0),1,src.size(1),src.size(2)).to(device)

        # print(src.shape)
        # src=src.transpose(0,1).to(DEVICE)
        # print(src)
        # inf_count = torch.sum(torch.eq(src, -float('inf'))).item()
        # print(inf_count)
        # src=src.long()
        x = self.word_embeddings(src).to(device)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # emb = torch.cat((x1, x2), dim=2).to(device)

        # inf_count = torch.sum(torch.eq(emb, 0)).item()
        # print(inf_count)
        # x = self.w_q(emb)
        # 位置信息
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # token编码+位置编码
        src = self.dropout(x + self.pos_embedding(pos))
        # print(src)
        src = src.transpose(1, 2).to(DEVICE)
        src = self.pool2(src).to(device)
        src = self.pool(src).to(device)
        # src=src.view(src.size(0),src.size(2),src.size(3)).to(device)
        # src=self.w_q2(src).to(device)
        src = src.transpose(1, 2).to(DEVICE)
        i = 0
        kl = 0
        for layer in self.layers:
            if i == 0:

                src, biaW = layer(src, src_mask)
                pos = src
            else:
                src, biaW2 = layer(src, src_mask)  # [batch_size, src_len, hid_dim]
                # kl = Syntacticdependency(yu, biaW2)
            i += 1
        return src, kl
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

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)  # [batch_size, query_len, n_heads, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            energy = energy.mask_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.dropout(attention), V)  # [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)  # [batch_size, query_len, hid_dim]

        x = self.fc_o(x)  # [batch_size, query_len, hid_dim]

        return x, energy
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
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, intent_size, slot_size, src_pad_idx):
        super(BERT, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, src_pad_idx)
        self.gelu = nn.GELU()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Dropout(dropout), nn.Tanh(), nn.LayerNorm(hid_dim))
        self.intent_out = nn.Linear(hid_dim, intent_size)
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.mlp = Model(hid_dim, hid_dim * 4, slot_size).to(device)

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]

    def forward(self, src, yu):
        # src_mask = self.make_src_mask(src)
        src_mask = 1
        encoder_out, kl = self.encoder(src, src_mask, yu)  # [batch_size, src_len, hid_dim]

        # 拿到[cls] token进行意图分类
        cls_hidden = self.fc(encoder_out[:, 0])  # [batch_size, hid_dim]
        intent_output = self.intent_out(cls_hidden)  # [batch_size, intent_size]
        intent_output = Fn.softmax(intent_output, dim=-1)

        return intent_output, kl