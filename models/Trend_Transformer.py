import torch
import torch.nn as nn
from module.encoders import GTrendEmbedder, PositionalEncoding
from module.decoders import TransformerDecoderLayer
from module.base_forecast import BaseForecastModel


class TrendTransformer(BaseForecastModel):
    def __init__(self, hidden_dim, output_dim, num_heads, num_layers, trend_len, num_trends, gpu_num, use_encoder_mask=1, dropout=0.3, rescale_value=1820):
        super().__init__(rescale_value=rescale_value)
        self.hidden_dim = hidden_dim
        self.output_len = output_dim
        self.save_hyperparameters()
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.horizon_query = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.02)
        self.horizon_pos_encoder = PositionalEncoding(hidden_dim, max_len=output_dim)
        self.horizon_head_weight = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.horizon_head_bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.horizon_head_weight)
        self.final_act = nn.Softplus()

    def forward(self, gtrends):
        gtrend_memory = self.gtrend_encoder(gtrends)
        batch_size = gtrends.size(0)
        tgt = self.horizon_query.unsqueeze(1).expand(self.output_len, batch_size, self.hidden_dim)
        tgt = self.horizon_pos_encoder(tgt)
        out = self.decoder(tgt, gtrend_memory)
        out_bt = out.permute(1, 0, 2)
        forecast_raw = (out_bt * self.horizon_head_weight.unsqueeze(0)).sum(dim=-1)
        forecast_raw = forecast_raw + self.horizon_head_bias.unsqueeze(0)
        forecast = self.final_act(forecast_raw)
        return forecast.clamp_min(0.0)

    def _predict_from_batch(self, batch):
        return self.forward(batch[2])