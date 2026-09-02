import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        d, w, m= temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), temporal_features[:, 2].unsqueeze(1)
        d_emb, w_emb, m_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(m)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb], dim=1))
        return self.dropout(temporal_embeddings)

class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.3)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon, device):
        mask = torch.zeros((size, size), device=device)
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i + split, i:i + split] = 1
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0, 2, 1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1, 0, 2))
        if self.use_mask == 1:
            input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon, gtrend_emb.device)
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb