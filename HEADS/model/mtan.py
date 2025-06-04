import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)
    
class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out  
    
class mTAN_Model(nn.Module):
    def __init__(self, args, enc_mtan_rnn, dec_mtan_rnn, input_dim, device):
        super().__init__()
        self.args = args
        self.device = device
        
        # 初始化时间参考点
        self.time_points = torch.linspace(0, 1., args.num_ref_points).to(device)
        
        # 初始化编码器和解码器
        self.enc = enc_mtan_rnn(
            input_dim,
            self.time_points,
            args.latent_dim,
            args.rec_hidden,
            embed_time=args.embed_time,
            learn_emb=args.learn_emb,
            num_heads=args.enc_num_heads
        ).to(device)

        self.dec = dec_mtan_rnn(
            input_dim,  # 原始数据维度
            self.time_points,
            args.latent_dim,
            args.gen_hidden,
            embed_time=args.embed_time,
            learn_emb=args.learn_emb,
            num_heads=args.dec_num_heads
        ).to(device)

        self.plugin = args.plugin
        if(self.plugin):
            self.epsilon = 1e-6

    def forward(self, x_data, x_mask, x_tp, tp_to_predict):
        # 拼接数据和掩码
        combined_input = torch.cat((x_data, x_mask), dim=2)
        
        # 编码器前向传播
        out = self.enc(combined_input, x_tp)
        qz0_mean = out[:, :, :self.args.latent_dim]
        qz0_logvar = out[:, :, self.args.latent_dim:]
        
        # 生成潜在变量
        epsilon = torch.randn(
            self.args.k_iwae, *qz0_mean.size(), device=self.device
        )
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, *qz0_mean.shape[1:])  # (k_iwae*bs, seq_len, latent_dim)
        
        # 解码器前向传播
        tp_to_predict_repeated = tp_to_predict[None, :, :].repeat(self.args.k_iwae, 1, 1)
        tp_to_predict_repeated = tp_to_predict_repeated.view(-1, tp_to_predict.shape[1])
        pred_x = self.dec(z0, tp_to_predict_repeated)
        
        # 调整输出形状
        batch_size = x_data.size(0)
        pred_x = pred_x.view(self.args.k_iwae, batch_size, *pred_x.shape[1:])
        
        return pred_x
    
    def forecasting(self, batch_dict):
        y_time, x_vals, x_time, x_mask, y_mask, y_vals = \
            batch_dict["tp_to_predict"].squeeze(0), \
            batch_dict["observed_data"].squeeze(0), \
            batch_dict["observed_tp"].squeeze(0), \
            batch_dict["observed_mask"].squeeze(0), \
            batch_dict["mask_predicted_data"].squeeze(0),\
            batch_dict["data_to_predict"].squeeze(0),\

        combined_input = torch.cat((x_vals, x_mask), dim=-1)
        
        out = self.enc(combined_input, x_time)
        qz0_mean = out[:, :, :self.args.latent_dim]
        qz0_logvar = out[:, :, self.args.latent_dim:]
        
        epsilon = torch.randn(
            self.args.k_iwae, *qz0_mean.size(), device=self.device
        )
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, *qz0_mean.shape[1:])  # (k_iwae*bs, seq_len, latent_dim)
        
        y_time_repeated = y_time[None, :, :].repeat(self.args.k_iwae, 1, 1)
        y_time_repeated = y_time_repeated.view(-1, y_time.shape[1])
        pred_y = self.dec(z0, y_time_repeated)
        
        batch_size = x_vals.size(0)
        pred_y = pred_y.view(self.args.k_iwae, batch_size, *pred_y.shape[1:])
        
        return pred_y, y_vals, y_mask