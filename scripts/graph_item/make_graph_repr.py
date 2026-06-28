import os, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_numpy_1d(x):
    x = x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    return x.reshape(-1).astype(np.float32)

def load_embedding_dict(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return {str(k).strip(): to_numpy_1d(v) for k, v in obj.items()}

class GatedFusion(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.gate_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, x_text, x_image, image_mask):
        x_text = F.normalize(x_text, p=2, dim=-1)
        x_image = F.normalize(x_image, p=2, dim=-1)
        t, v = self.text_proj(x_text), self.image_proj(x_image)
        g = self.gate_mlp(torch.cat([t, v, image_mask], dim=-1))
        return image_mask * (g * t + (1.0 - g) * v) + (1.0 - image_mask) * t

class StudentEncoder(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=256, out_dim=256, dropout=0.2):
        super().__init__()
        self.fusion = GatedFusion(text_dim, image_dim, hidden_dim, dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.skip_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_text, x_image, image_mask):
        h0 = self.fusion(x_text, x_image, image_mask)
        h = self.mlp(h0)
        return F.normalize(self.out_proj(h) + self.skip_proj(h0), p=2, dim=-1)

def build_rows(df, text_dict, image_dict, key_col):
    rows = []
    for key in df[key_col].astype(str):
        rows.append((key, text_dict[key], image_dict[key]))
    return rows

@torch.no_grad()
def generate_embeddings(model, rows, device, batch_size):
    model.eval()
    keys, embs = [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        x_text = torch.tensor(np.stack([r[1] for r in batch]), dtype=torch.float, device=device)
        x_image = torch.tensor(np.stack([r[2] for r in batch]), dtype=torch.float, device=device)
        image_mask = torch.ones((len(batch), 1), dtype=torch.float, device=device)
        z = model(x_text, x_image, image_mask).cpu()
        keys.extend([r[0] for r in batch])
        embs.append(z)
    return keys, torch.cat(embs, dim=0)

def save_embeddings(path, keys, embs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({k: e.numpy() for k, e in zip(keys, embs)}, f)

def main(args):
    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() and args.gpu_num >= 0 else "cpu"
    df = pd.read_csv(args.item_csv_path)
    text_dict = load_embedding_dict(args.text_emb_path)
    image_dict = load_embedding_dict(args.image_emb_path)
    rows = build_rows(df, text_dict, image_dict, args.key_col)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    config = ckpt["config"]
    model = StudentEncoder(config["text_dim"], config["image_dim"], config["hidden_dim"], config["out_dim"], config["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    keys, embs = generate_embeddings(model, rows, device, args.batch_size)
    save_path = os.path.join(args.save_dir, args.output_name)
    save_embeddings(save_path, keys, embs)
    print(f"items={len(keys)}, dim={embs.size(1)}")
    print(f"saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--item_csv_path", type=str, required=True)
    parser.add_argument("--text_emb_path", type=str, required=True)
    parser.add_argument("--image_emb_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/item_embeddings")
    parser.add_argument("--output_name", type=str, default="graph_item_emb.pkl")
    parser.add_argument("--key_col", type=str, default="item_number_color")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    main(args)