import torch
import torch.nn as nn
from module.encoders import DummyEmbedder
from module.base_forecast import BaseForecastModel


class StaticMLPBaseline(BaseForecastModel):
    def __init__(self, hidden_dim, output_dim, use_item_branch=False, use_img_branch=True, use_text_branch=True, use_temporal=True, dropout=0.3, rescale_value=1820):
        super().__init__(rescale_value=rescale_value)
        self.hidden_dim = hidden_dim
        self.output_len = output_dim
        self.use_item_branch = use_item_branch
        self.use_img_branch = use_img_branch
        self.use_text_branch = use_text_branch
        self.use_temporal = use_temporal
        self.save_hyperparameters()
        
        if self.use_item_branch:
            self.item_proj = nn.Linear(128, hidden_dim)
        if self.use_img_branch:
            self.image_proj = nn.Linear(512, hidden_dim)
        if self.use_text_branch:
            self.text_proj = nn.Linear(512, hidden_dim)
        if self.use_temporal:
            self.dummy_encoder = DummyEmbedder(hidden_dim)
            
        input_dim = hidden_dim * (int(use_item_branch) + int(use_img_branch) + int(use_text_branch) + int(use_temporal))
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim), 
            nn.Linear(input_dim, hidden_dim * 2, bias=False), 
            nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(hidden_dim * 2, output_dim)
        )
        self.final_act = nn.Softplus()

    def forward(self, temporal_features, item_embedding, img_embs, text_embs):
        feats = []
        if self.use_item_branch:
            feats.append(self.item_proj(item_embedding.to(self.device).float()))
        if self.use_img_branch:
            feats.append(self.image_proj(img_embs.to(self.device).float()))
        if self.use_text_branch:
            feats.append(self.text_proj(text_embs.to(self.device).float()))
        if self.use_temporal:
            feats.append(self.dummy_encoder(temporal_features))
        forecast = self.final_act(self.mlp(torch.cat(feats, dim=1)))
        return forecast.clamp_min(0.0)

    def _predict_from_batch(self, batch):
        return self.forward(batch[1], batch[3], batch[4], batch[5])