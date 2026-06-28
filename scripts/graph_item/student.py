import os, random, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_numpy_1d(x):
    x = x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    return x.reshape(-1).astype(np.float32)

def load_multimodal_embedding_dicts(path):
    obj = torch.load(path, map_location="cpu")
    asins, text_embs, img_embs = obj["asins"], obj["text_embeddings"], obj["image_embeddings"]
    text_dict, image_dict = {}, {}
    for a, t, i in zip(asins, text_embs, img_embs):
        text_dict[str(a)] = to_numpy_1d(t)
        image_dict[str(a)] = to_numpy_1d(i)
    return text_dict, image_dict

def load_teacher_embedding_dict(path):
    obj = torch.load(path, map_location="cpu")
    if "embeddings" not in obj or "asins" not in obj:
        raise ValueError("teacher embedding file must include embeddings and asins")
    embs = obj["embeddings"].detach().cpu().float().numpy() if isinstance(obj["embeddings"], torch.Tensor) else np.asarray(obj["embeddings"], dtype=np.float32)
    asins = obj["asins"]
    if len(asins) != len(embs):
        raise ValueError("teacher embedding length mismatch")
    return {str(a): np.asarray(e, dtype=np.float32) for a, e in zip(asins, embs)}

class StudentDistillDataset(Dataset):
    def __init__(self, text_dict, image_dict, teacher_dict):
        self.rows = []
        img_dim = len(next(iter(image_dict.values())))
        skipped_text, skipped_teacher, missing_img = 0, 0, 0
        for asin, z in teacher_dict.items():
            if asin not in text_dict:
                skipped_text += 1; continue
            if z is None:
                skipped_teacher += 1; continue
            if asin in image_dict:
                img, mask = image_dict[asin].astype(np.float32), 1.0
            else:
                img, mask = np.zeros(img_dim, dtype=np.float32), 0.0; missing_img += 1
            self.rows.append((asin, text_dict[asin].astype(np.float32), img, mask, z.astype(np.float32)))
        print(f"[Dataset] n={len(self.rows)}, skipped_text={skipped_text}, skipped_teacher={skipped_teacher}, missing_img={missing_img}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        asin, text, image, mask, target = self.rows[idx]
        return {"asin": asin, "x_text": torch.tensor(text).float(), "x_image": torch.tensor(image).float(), "image_mask": torch.tensor([mask]).float(), "target": torch.tensor(target).float()}

def split_dataset(dataset, val_ratio):
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    train_ds = StudentDistillDataset.__new__(StudentDistillDataset)
    val_ds = StudentDistillDataset.__new__(StudentDistillDataset)
    train_ds.rows = [dataset.rows[i] for i in idx[n_val:]]
    val_ds.rows = [dataset.rows[i] for i in idx[:n_val]]
    return train_ds, val_ds

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

def load_teacher_fusion_weights(model, ckpt_path, freeze=False):
    if ckpt_path is None or ckpt_path == "":
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    fusion_state = {}
    for k, v in state.items():
        if k.startswith("fusion.fusion."):
            fusion_state[k.replace("fusion.fusion.", "")] = v
        elif k.startswith("fusion."):
            fusion_state[k.replace("fusion.", "")] = v
    missing, unexpected = model.fusion.load_state_dict(fusion_state, strict=False)
    print(f"[Load Teacher Fusion] loaded={len(fusion_state)}, missing={missing}, unexpected={unexpected}")
    if freeze:
        for p in model.fusion.parameters():
            p.requires_grad = False

def batch_kl_loss(student_z, teacher_z, tau):
    student_z = F.normalize(student_z, p=2, dim=-1)
    teacher_z = F.normalize(teacher_z, p=2, dim=-1)
    s_logits = torch.matmul(student_z, student_z.T) / tau
    t_logits = torch.matmul(teacher_z, teacher_z.T) / tau
    mask = torch.eye(s_logits.size(0), dtype=torch.bool, device=s_logits.device)
    s_logits = s_logits.masked_fill(mask, -1e9)
    t_logits = t_logits.masked_fill(mask, -1e9)
    return F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1), reduction="batchmean")

def distill_loss(student_z, teacher_z, args):
    teacher_z = F.normalize(teacher_z, p=2, dim=-1)
    student_z = F.normalize(student_z, p=2, dim=-1)
    cos = (1.0 - F.cosine_similarity(student_z, teacher_z, dim=-1)).mean()
    mse = F.mse_loss(student_z, teacher_z)
    kl = batch_kl_loss(student_z, teacher_z, args.distill_tau)
    return args.lambda_cos * cos + args.lambda_mse * mse + args.lambda_kl * kl, cos, mse, kl

@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total = np.zeros(4, dtype=np.float64)
    cnt = 0
    for batch in loader:
        x_text = batch["x_text"].to(device)
        x_image = batch["x_image"].to(device)
        image_mask = batch["image_mask"].to(device)
        target = batch["target"].to(device)
        pred = model(x_text, x_image, image_mask)
        loss, cos, mse, kl = distill_loss(pred, target, args)
        bsz = x_text.size(0)
        total += np.array([loss.item(), cos.item(), mse.item(), kl.item()]) * bsz
        cnt += bsz
    return total / max(cnt, 1)

@torch.no_grad()
def generate_embeddings(model, dataset, device, batch_size):
    model.eval()
    asins, embs = [], []
    for i in range(0, len(dataset.rows), batch_size):
        rows = dataset.rows[i:i + batch_size]
        x_text = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float, device=device)
        x_image = torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.float, device=device)
        image_mask = torch.tensor(np.array([[r[3]] for r in rows], dtype=np.float32), dtype=torch.float, device=device)
        z = model(x_text, x_image, image_mask).cpu()
        asins.extend([r[0] for r in rows])
        embs.append(z)
    return asins, torch.cat(embs, dim=0)

def save_ckpt(path, model, text_dim, image_dim, args):
    torch.save({"model_state_dict": model.state_dict(), "config": {"text_dim": text_dim, "image_dim": image_dim, "hidden_dim": args.hidden_dim, "out_dim": args.out_dim, "dropout": args.dropout}}, path)

def train(args):
    set_seed(args.seed)
    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() and args.gpu_num >= 0 else "cpu"
    text_dict, image_dict = load_multimodal_embedding_dicts(args.fclip_emb_path)
    teacher_dict = load_teacher_embedding_dict(args.teacher_emb_path)
    text_dim = len(next(iter(text_dict.values())))
    image_dim = len(next(iter(image_dict.values())))
    teacher_dim = len(next(iter(teacher_dict.values())))

    full_ds = StudentDistillDataset(text_dict, image_dict, teacher_dict)
    train_ds, val_ds = split_dataset(full_ds, args.val_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"[Config] device={device}, train={len(train_ds)}, val={len(val_ds)}, text_dim={text_dim}, image_dim={image_dim}, teacher_dim={teacher_dim}")
    model = StudentEncoder(text_dim, image_dim, args.hidden_dim, args.out_dim, args.dropout).to(device)
    load_teacher_fusion_weights(model, args.teacher_encoder_ckpt, args.freeze_teacher_fusion)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)

    out_dir = os.path.join(args.save_dir, datetime.now().strftime("%y%m%d_%H%M_student"))
    os.makedirs(out_dir, exist_ok=True)
    best_val, best_path, prev_best = float("inf"), None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = np.zeros(4, dtype=np.float64)
        train_cnt = 0

        for batch in train_loader:
            x_text = batch["x_text"].to(device, non_blocking=True)
            x_image = batch["x_image"].to(device, non_blocking=True)
            image_mask = batch["image_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x_text, x_image, image_mask)
            loss, cos, mse, kl = distill_loss(pred, target, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            bsz = x_text.size(0)
            train_sum += np.array([loss.item(), cos.item(), mse.item(), kl.item()]) * bsz
            train_cnt += bsz

        train_loss, train_cos, train_mse, train_kl = train_sum / max(train_cnt, 1)
        val_loss, val_cos, val_mse, val_kl = evaluate(model, val_loader, device, args)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            if prev_best and os.path.exists(prev_best):
                os.remove(prev_best)
            best_path = os.path.join(out_dir, f"best_student_encoder_{epoch}.pt")
            save_ckpt(best_path, model, text_dim, image_dim, args)
            prev_best = best_path

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} train_cos={train_cos:.6f} train_mse={train_mse:.6f} train_kl={train_kl:.6f} val_loss={val_loss:.6f} val_cos={val_cos:.6f} val_mse={val_mse:.6f} val_kl={val_kl:.6f}")

    last_path = os.path.join(out_dir, "last_student_encoder.pt")
    save_ckpt(last_path, model, text_dim, image_dim, args)
    best_path = best_path or last_path

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    asins, embs = generate_embeddings(model, full_ds, device, args.gen_batch_size)
    emb_path = os.path.join(out_dir, "student_item_embeddings.pt")
    torch.save({"embeddings": embs, "asins": asins, "dim": embs.size(1)}, emb_path)
    print(f"Best student saved to: {best_path}")
    print(f"Saved student embeddings to: {emb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fclip_emb_path", type=str, required=True)
    parser.add_argument("--teacher_emb_path", type=str, required=True)
    parser.add_argument("--teacher_encoder_ckpt", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./outputs/graph_item")
    parser.add_argument("--freeze_teacher_fusion", action="store_true")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gen_batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lambda_cos", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.05)
    parser.add_argument("--lambda_kl", type=float, default=0.7)
    parser.add_argument("--distill_tau", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)
    args = parser.parse_args()
    train(args)