import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers.optimization import Adafactor


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

        x_reshape = x.contiguous().view(-1, x.size(-1))  
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
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
    
class NeighborRawSignalEncoder(nn.Module):
    def __init__(self, hidden_dim, input_len=12, time_feat_dim=4, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        self.time_feat_dim = time_feat_dim

        self.input_proj = nn.Linear(1 + time_feat_dim, hidden_dim)
        self.pos_embedding = PositionalEncoding(hidden_dim, dropout=dropout, max_len=input_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        self.sales_signal_proj = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.sim_scale = nn.Parameter(torch.tensor(0.1))                
        self.dropout = nn.Dropout(dropout)

    def forward(self, neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid, base_context):
        B, K, T = neighbor_sales.shape

        neighbor_sales = torch.nan_to_num(neighbor_sales, nan=0.0, posinf=0.0, neginf=0.0)
        neighbor_mask = torch.nan_to_num(neighbor_mask, nan=0.0, posinf=0.0, neginf=0.0)
        neighbor_time_feats = torch.nan_to_num(neighbor_time_feats, nan=0.0, posinf=0.0, neginf=0.0)
        neighbor_sims = torch.nan_to_num(neighbor_sims, nan=0.0, posinf=0.0, neginf=0.0)
        neighbor_valid = torch.nan_to_num(neighbor_valid, nan=0.0, posinf=0.0, neginf=0.0)
        
        valid_count = neighbor_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_level = (neighbor_sales * neighbor_mask).sum(dim=-1, keepdim=True) / valid_count
        
        neighbor_sales_shape = neighbor_sales / neighbor_level.clamp_min(1e-6)
        neighbor_sales_shape = torch.nan_to_num(
            neighbor_sales_shape,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        neighbor_sales_shape = neighbor_sales_shape * neighbor_mask

        x = torch.cat([neighbor_sales_shape.unsqueeze(-1), neighbor_time_feats,],dim=-1)
        x = x * neighbor_mask.unsqueeze(-1)
        x = self.input_proj(x)
        x = x.view(B * K, T, self.hidden_dim)

        seq_valid = neighbor_mask.view(B * K, T) > 0
        key_padding_mask = ~seq_valid

        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask[all_invalid, 0] = False
            x[all_invalid, 0, :] = 0.0

        x = x.transpose(0, 1)
        x = self.pos_embedding(x)
        x = self.seq_encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)

        valid_float = (~key_padding_mask).float().unsqueeze(-1)  # [B*K, T, 1]
        neighbor_vec = (x * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        neighbor_vec = neighbor_vec.view(B, K, self.hidden_dim)  # [B, K, H]
        neighbor_vec = self.dropout(neighbor_vec)

        week_valid_count = neighbor_mask.sum(dim=-1)  # [B, K]
        neighbor_valid_mask = (neighbor_valid > 0) & (week_valid_count > 0)

        valid_neighbor_float = neighbor_valid_mask.float()  # [B, K]
        valid_week_mask = neighbor_mask * valid_neighbor_float.unsqueeze(-1)
        valid_neighbor_count = valid_neighbor_float.sum(dim=1, keepdim=True).clamp_min(1.0)

        signal_valid_ratio = valid_week_mask.sum(dim=(1, 2)).unsqueeze(1) / (valid_neighbor_count * T)
        signal_valid_ratio = signal_valid_ratio.clamp(0.0, 1.0)

        # attention aggregation conditioned on base_context
        q = self.query_proj(base_context).unsqueeze(1)  # [B, 1, H]
        k = self.key_proj(neighbor_vec)                 # [B, K, H]

        learned_scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)
        sim_scores = neighbor_sims
        sim_bias = sim_scores - sim_scores.mean(dim=1, keepdim=True)
        scores = learned_scores + self.sim_scale * sim_bias
        scores = scores.masked_fill(~neighbor_valid_mask, float("-inf"))

        all_invalid_item = ~neighbor_valid_mask.any(dim=1)
        scores = torch.where(all_invalid_item.unsqueeze(1), torch.zeros_like(scores), scores)

        valid_item_mask = (~all_invalid_item).float().unsqueeze(1)
        attn_weights = torch.softmax(scores, dim=1) * valid_item_mask
        weighted_mask = attn_weights.unsqueeze(-1) * neighbor_mask 

        # make final signal
        denom = weighted_mask.sum(dim=1)
        neighbor_signal = (weighted_mask * neighbor_sales_shape).sum(dim=1) / denom.clamp_min(1e-8)

        neighbor_signal_mask = (denom > 0).float()
        neighbor_signal = neighbor_signal * neighbor_signal_mask
        neighbor_context = self.sales_signal_proj(neighbor_signal)

        return neighbor_context, signal_valid_ratio, attn_weights
    
class StaticTemporalFusion(nn.Module):
    def __init__(self, hidden_dim, use_item_branch=True, use_img_branch=True, use_text_branch=True, dropout=0.2):
        super().__init__()

        self.use_item_branch = use_item_branch
        self.use_img_branch = use_img_branch
        self.use_text_branch = use_text_branch
        
        self.aux_dim = 16
        self.gate_scale = 0.3
        
        self.img_proj = nn.Sequential(
            nn.Linear(512, self.aux_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.aux_dim, hidden_dim, bias=False),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(512, self.aux_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.aux_dim, hidden_dim, bias=False),
        )

        if self.use_item_branch:

            if self.use_img_branch:
                self.img_gate = nn.Linear(hidden_dim * 2, 1)
                nn.init.constant_(self.img_gate.bias, -2.0)
                
            if self.use_text_branch:
                self.text_gate = nn.Linear(hidden_dim * 2, 1)
                nn.init.constant_(self.text_gate.bias, -2.0)

            self.item_proj = nn.Linear(128, hidden_dim)
            self.item_aux_norm = nn.LayerNorm(hidden_dim)
            
            self.final_fusion = nn.Sequential(
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        else:
            num_inputs = 1

            if self.use_img_branch:
                num_inputs += 1
            if self.use_text_branch:
                num_inputs += 1

            input_dim = hidden_dim * num_inputs

            self.fusion = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

    def forward(self, static_temporal, static_item=None, static_image=None, static_text=None):
        
        if self.use_img_branch:
            static_image = self.img_proj(static_image)
        if self.use_text_branch:
            static_text = self.text_proj(static_text)
        
        # case 1) item branch ON: item-centric gated residual
        if self.use_item_branch:
            static_item = self.item_proj(static_item)
            item_imgtxt_fused = static_item
            img_gate, text_gate = None, None

            if self.use_img_branch:
                img_gate = self.gate_scale * torch.sigmoid(self.img_gate(torch.cat([static_item, static_image], dim=1)))
                item_imgtxt_fused = item_imgtxt_fused + img_gate * static_image
                
            if self.use_text_branch:
                text_gate = self.gate_scale * torch.sigmoid(self.text_gate(torch.cat([static_item, static_text], dim=1))) 
                item_imgtxt_fused = item_imgtxt_fused + text_gate * static_text
            
            if self.use_img_branch and self.use_text_branch:
                item_imgtxt_fused = self.item_aux_norm(item_imgtxt_fused)

            x = torch.cat([item_imgtxt_fused, static_temporal], dim=1)
            static_fused = self.final_fusion(x)

            return static_fused, img_gate, text_gate

        # case 2) item branch OFF: concat fusion
        else:
            feats = []

            if self.use_img_branch:
                feats.append(static_image)

            if self.use_text_branch:
                feats.append(static_text)

            feats.append(static_temporal)

            x = torch.cat(feats, dim=1)
            static_fused = self.fusion(x)

            return static_fused, None, None

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

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

class GraphForecast(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, trend_len, num_trends, gpu_num, 
                 use_encoder_mask=1, use_graph_item=True, use_text=True, use_img=True, use_signal=False, rescale_val=1065):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_graph_item = use_graph_item
        self.use_text = use_text
        self.use_img = use_img
        self.use_signal = use_signal
        self.rescale_val = rescale_val
        self.save_hyperparameters()

        # Encoder
        self.dummy_encoder = DummyEmbedder(hidden_dim)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.neighbor_set_encoder = NeighborRawSignalEncoder(
            hidden_dim=hidden_dim,
            input_len=output_dim,
            time_feat_dim=4,
            num_heads=num_heads,
            dropout=0.3,
        )
        self.static_fusion = StaticTemporalFusion(
            hidden_dim=hidden_dim,
            use_item_branch=self.use_graph_item,
            use_img_branch=self.use_img,
            use_text_branch=self.use_text,
            dropout=0.3,
        )
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.3,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.horizon_query = nn.Parameter(torch.randn(self.output_len, hidden_dim) * 0.02)
        self.horizon_pos_encoder = PositionalEncoding(hidden_dim, max_len=self.output_len)
        
        self.tgt_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.horizon_head_weight = nn.Parameter(torch.empty(self.output_len, hidden_dim))
        self.horizon_head_bias = nn.Parameter(torch.zeros(self.output_len))
        nn.init.xavier_uniform_(self.horizon_head_weight)
        self.final_act = nn.Softplus()
        
        # Correction
        self.signal_correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.signal_shape_scale = 0.8

    def forward(self, temporal_features, gtrends, item_embedding, img_embs, text_embs, 
                neighbor_sales=None, neighbor_mask=None, neighbor_time_feats=None, neighbor_sims=None, neighbor_valid=None):

        # Encode
        gtrend_memory = self.gtrend_encoder(gtrends)
        batch_size = gtrends.size(0)

        static_item = None
        if self.use_graph_item:
            static_item = item_embedding.to(self.device).float()

        static_image = None
        if self.use_img:
            static_image = img_embs.to(self.device).float()

        static_text = None
        if self.use_text:
            static_text = text_embs.to(self.device).float()     

        static_temporal = self.dummy_encoder(temporal_features)

        static_fused, _, _ = self.static_fusion(
            static_temporal=static_temporal,
            static_item=static_item,
            static_image=static_image,
            static_text=static_text,
        )                                    

        # Decode
        base_tgt = self.horizon_query.unsqueeze(1).expand(self.output_len, batch_size, self.hidden_dim)    
        base_tgt = self.horizon_pos_encoder(base_tgt)
        static_expand = static_fused.unsqueeze(0).expand(self.output_len, -1, -1)
        tgt = self.tgt_fusion(torch.cat([base_tgt, static_expand], dim=-1))
        
        out, attn_weights = self.decoder(tgt, gtrend_memory)        
        out_bt = out.permute(1, 0, 2)              
        decoder_context = out.mean(dim=0)

        forecast_raw = (out_bt * self.horizon_head_weight.unsqueeze(0)).sum(dim=-1)
        forecast_raw = forecast_raw + self.horizon_head_bias.unsqueeze(0)
        forecast = self.final_act(forecast_raw)

        # Correction
        if self.use_signal:
            signal_base_context = decoder_context.detach()
            neighbor_context, signal_valid_ratio, _ = self.neighbor_set_encoder(
                neighbor_sales=neighbor_sales,
                neighbor_mask=neighbor_mask,
                neighbor_time_feats=neighbor_time_feats,
                neighbor_sims=neighbor_sims,
                neighbor_valid=neighbor_valid,
                base_context=signal_base_context,
            )

            signal_delta = torch.tanh(self.signal_correction_head(neighbor_context))
            signal_delta = signal_delta - signal_delta.mean(dim=1, keepdim=True)

            signal_mul = self.signal_shape_scale
            signal_factor = 1.0 + signal_mul * signal_valid_ratio * signal_delta
            signal_factor = signal_factor.clamp(1-signal_mul, 1+signal_mul)
            signal_factor = signal_factor / signal_factor.mean(dim=1, keepdim=True).clamp_min(1e-6)

            forecast = forecast * signal_factor

        forecast = forecast.clamp_min(0.0)

        return forecast, attn_weights

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        item_sales, temporal_features, gtrends, items, img_embs, text_embs, \
        neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid = train_batch
        
        if not self.use_signal:
            neighbor_sales = neighbor_mask = neighbor_time_feats = neighbor_sims = neighbor_valid = None

        forecasted_sales, _ = self.forward(temporal_features, gtrends, items, img_embs, text_embs,
            neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid)
        
        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log_dict("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, test_batch, batch_idx):
        item_sales, temporal_features, gtrends, items, img_embs, text_embs, \
        neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid = test_batch
        
        if not self.use_signal:
            neighbor_sales = neighbor_mask = neighbor_time_feats = neighbor_sims = neighbor_valid = None

        forecasted_sales, _ = self.forward(temporal_features, gtrends, items, img_embs, text_embs,
            neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid)

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