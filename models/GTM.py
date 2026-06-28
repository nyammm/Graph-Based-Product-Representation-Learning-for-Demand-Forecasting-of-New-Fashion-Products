import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import pipeline
from transformers.optimization import Adafactor
from torchvision import models


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
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super(FusionNetwork, self).__init__()
        
        # self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_linear = nn.Linear(512, embedding_dim) 
        self.text_linear = nn.Linear(512, embedding_dim)
        
        input_dim = embedding_dim + (embedding_dim*2)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        # Fuse static features together
        condensed_img = self.img_linear(img_encoding.squeeze(1))
        condensed_text = self.text_linear(text_encoding.squeeze(1))

        # Build input
        decoder_inputs = []
        decoder_inputs.append(condensed_img) 
        decoder_inputs.append(condensed_text) 
        decoder_inputs.append(dummy_encoding)
        concat_features = torch.cat(decoder_inputs, dim=1)

        final = self.feature_fusion(concat_features)

        return final

class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends,  gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask
    
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0,2,1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1,0,2))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon)
        if self.use_mask == 1:
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim*3, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month)
        d, w, m = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), \
            temporal_features[:, 2].unsqueeze(1)
        d_emb, w_emb, m_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(m)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, 
            memory_key_padding_mask = None):

        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

class GTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, \
                trend_len, num_trends, gpu_num, use_encoder_mask=1, autoregressive=False, rescale_val=1065):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.gpu_num = gpu_num
        self.rescale_val = rescale_val
        self.save_hyperparameters()

         # Encoder
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim)

        # Decoder
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, \
                                                dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        
        if self.autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not self.autoregressive else 1),
            nn.Softplus(),
            nn.Dropout(0.2)
        )
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, temporal_features, gtrends, image_embs, text_embs):
        # Encode features and get inputs
        img_encoding = image_embs.to(self.device)
        text_encoding = text_embs.to(self.device)
        dummy_encoding = self.dummy_encoder(temporal_features)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)

        if self.autoregressive == 1:
            # Decode
            tgt = torch.zeros(self.output_len, gtrend_encoding.shape[1], gtrend_encoding.shape[-1]).to('cuda:'+str(self.gpu_num))
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory, tgt_mask)
            forecast = self.decoder_fc(decoder_out)
        else:
            # Decode (generatively/non-autoregressively)
            tgt = static_feature_fusion.unsqueeze(0)
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory)
            forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        item_sales, temporal_features, gtrends, img_embs, text_embs = train_batch
        forecasted_sales, _ = self.forward(temporal_features, gtrends, img_embs, text_embs)
        
        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log_dict("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, test_batch, batch_idx):
        item_sales, temporal_features, gtrends, img_embs, text_embs = test_batch
        forecasted_sales, _ = self.forward(temporal_features, gtrends, img_embs, text_embs)

        return {"y": item_sales, "pred": forecasted_sales}

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.cat([x["y"] for x in val_step_outputs], dim=0)
        forecasted_sales = torch.cat([x["pred"] for x in val_step_outputs], dim=0)

        rescaled_item_sales = item_sales * self.rescale_val
        rescaled_forecasted_sales = forecasted_sales * self.rescale_val

        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)

        eps = 1e-8
        trend = rescaled_item_sales
        outputs = rescaled_forecasted_sales

        wape = torch.sum(torch.abs(trend - outputs)) / (torch.sum(torch.abs(trend)) + eps)

        smape_t = torch.abs(trend - outputs) / (torch.abs(trend) + torch.abs(outputs) + eps)
        adj_smape = smape_t.mean(dim=-1)
        adj_smape_mean = adj_smape.mean()

        sum_trend = trend.sum(dim=-1)
        sum_out = outputs.sum(dim=-1)
        accum_smape = torch.abs(sum_trend - sum_out) / (torch.abs(sum_trend) + torch.abs(sum_out) + eps)
        accum_smape_mean = accum_smape.mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_mae": mae,
                "val_adj_smape": adj_smape_mean,
                "val_accum_smape": accum_smape_mean,
                "val_wape": wape,
            },
            logger=True,
            on_epoch=True,
        )
