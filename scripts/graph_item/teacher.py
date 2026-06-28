import os, random, argparse
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


LEVEL_WEIGHT = {"strong": 1.0, "weak": 0.6, "loose": 0.2, "unknown": 0.4}
DEGREE_PENALTY_ETA = 0.2
MAX_POS_PER_ANCHOR = 20


def parse_levels(x):
    return set(v.strip() for v in x.split(",") if v.strip())

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_numpy_1d(x):
    x = x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    return x.reshape(-1).astype(np.float32)

def load_multimodal_embedding_dicts(path):
    obj = torch.load(path, map_location="cpu")
    asins, text_embs, img_embs = obj["asins"], obj["text_embeddings"], obj["image_embeddings"]
    text_embs = text_embs.detach().cpu().float().numpy() if isinstance(text_embs, torch.Tensor) else np.asarray(text_embs, dtype=np.float32)
    img_embs = img_embs.detach().cpu().float().numpy() if isinstance(img_embs, torch.Tensor) else np.asarray(img_embs, dtype=np.float32)
    if len(asins) != len(text_embs) or len(asins) != len(img_embs):
        raise ValueError("embedding length mismatch")
    return {str(a): to_numpy_1d(e) for a, e in zip(asins, text_embs)}, {str(a): to_numpy_1d(e) for a, e in zip(asins, img_embs)}

def build_anchor_groups(src_list, dst_list, w_list, max_pos):
    grouped = defaultdict(list)
    for s, d, w in zip(src_list, dst_list, w_list):
        grouped[s].append((d, w))
    anchors, offsets, nodes, weights = [], [0], [], []
    for a, pairs in grouped.items():
        anchors.append(a)
        for d, w in sorted(pairs, key=lambda x: x[1], reverse=True)[:max_pos]:
            nodes.append(d); weights.append(w)
        offsets.append(len(nodes))
    return torch.tensor(anchors).long(), torch.tensor(offsets).long(), torch.tensor(nodes).long(), torch.tensor(weights).float()

def build_k_hop_neighbors(edge_index, num_nodes, k_min=2, k_max=3):
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[s].append(d)
    outs = []
    for node in range(num_nodes):
        visited, q, result = {node}, deque([(node, 0)]), set()
        while q:
            cur, depth = q.popleft()
            if k_min <= depth <= k_max:
                result.add(cur)
            if depth >= k_max:
                continue
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt); q.append((nxt, depth + 1))
        result.discard(node); outs.append(sorted(result))
    return outs

def build_graph_data(items_path, pos_edges_path, multimodal_emb_path, use_graph_levels, use_loss_levels):
    items_df = pd.read_parquet(items_path)
    edges_df = pd.read_parquet(pos_edges_path)
    text_dict, img_dict = load_multimodal_embedding_dicts(multimodal_emb_path)
    graph_edges = edges_df[edges_df["pos_level"].isin(use_graph_levels)].copy()
    loss_edges = edges_df[edges_df["pos_level"].isin(use_loss_levels)].copy()
    used = set(graph_edges["src_asin"].astype(str)) | set(graph_edges["dst_asin"].astype(str))
    items_df = items_df[items_df["asin"].astype(str).isin(used)].sort_values("item_id").reset_index(drop=True)

    text_dim = next(iter(text_dict.values())).shape[0]
    img_dim = next(iter(img_dict.values())).shape[0]
    rows, texts, imgs, masks = [], [], [], []
    for _, row in items_df.iterrows():
        asin = str(row["asin"])
        if asin not in text_dict:
            continue
        rows.append(row); texts.append(text_dict[asin])
        if asin in img_dict:
            imgs.append(img_dict[asin]); masks.append([1.0])
        else:
            imgs.append(np.zeros(img_dim, dtype=np.float32)); masks.append([0.0])
    items_df = pd.DataFrame(rows).reset_index(drop=True)
    if len(items_df) == 0:
        raise ValueError("No valid nodes after embedding filtering")
    items_df["new_item_id"] = np.arange(len(items_df))
    asin2id = dict(zip(items_df["asin"].astype(str), items_df["new_item_id"]))

    x_text = torch.tensor(np.stack(texts), dtype=torch.float)
    x_image = torch.tensor(np.stack(imgs), dtype=torch.float)
    image_mask = torch.tensor(np.asarray(masks, dtype=np.float32), dtype=torch.float)

    graph_rows = []
    for _, r in graph_edges.iterrows():
        s, d = str(r["src_asin"]), str(r["dst_asin"])
        if s in asin2id and d in asin2id:
            w = 1.0 if pd.isna(r["pos_weight"]) else float(r["pos_weight"])
            graph_rows.append((asin2id[s], asin2id[d], w, r["pos_level"]))
    if len(graph_rows) == 0:
        raise ValueError("No available graph edges")

    base_deg = np.zeros(len(items_df), dtype=np.int64)
    for s, d, _, _ in graph_rows:
        base_deg[s] += 1; base_deg[d] += 1

    graph_src, graph_dst, graph_w = [], [], []
    for s, d, w, lvl in graph_rows:
        deg_penalty = 1.0 / ((max(base_deg[s], 1) * max(base_deg[d], 1)) ** (DEGREE_PENALTY_ETA / 2.0))
        ww = w * LEVEL_WEIGHT.get(lvl, 1.0) * deg_penalty
        graph_src += [s, d]; graph_dst += [d, s]; graph_w += [ww, ww]

    edge_index = torch.tensor([graph_src, graph_dst]).long()
    edge_weight = torch.tensor(graph_w).float()
    degree = np.bincount(edge_index[0].numpy(), minlength=len(items_df))
    degree_tensor = torch.tensor(degree).float()

    hop_list = build_k_hop_neighbors(edge_index, len(items_df), 2, 3)
    hop_flat, hop_offsets = [], [0]
    for ns in hop_list:
        hop_flat += ns; hop_offsets.append(len(hop_flat))
    hop_neighbors = torch.tensor(hop_flat).long()
    hop_offsets = torch.tensor(hop_offsets).long()

    pair_src, pair_dst, pair_w = [], [], []
    for _, r in loss_edges.iterrows():
        s, d = str(r["src_asin"]), str(r["dst_asin"])
        if s not in asin2id or d not in asin2id:
            continue
        src, dst = asin2id[s], asin2id[d]
        w = 1.0 if pd.isna(r["pos_weight"]) else float(r["pos_weight"])
        deg_penalty = 1.0 / ((max(degree[src], 1) * max(degree[dst], 1)) ** (DEGREE_PENALTY_ETA / 2.0))
        ww = w * LEVEL_WEIGHT.get(r["pos_level"], 1.0) * deg_penalty
        pair_src += [src, dst]; pair_dst += [dst, src]; pair_w += [ww, ww]
    if len(pair_src) == 0:
        raise ValueError("No available loss pairs")

    pair_index = torch.tensor([pair_src, pair_dst]).long()
    pair_weight = torch.tensor(pair_w).float()

    pos_dict = defaultdict(set)
    for s, d in zip(pair_src, pair_dst):
        pos_dict[int(s)].add(int(d))
    direct_neighbors, direct_offsets = [], [0]
    for i in range(len(items_df)):
        ns = sorted(pos_dict.get(i, set()))
        direct_neighbors += ns; direct_offsets.append(len(direct_neighbors))

    anchor_ids, offsets, pos_nodes, pos_weights = build_anchor_groups(pair_src, pair_dst, pair_w, MAX_POS_PER_ANCHOR)
    data = Data(x_text=x_text, x_image=x_image, image_mask=image_mask, edge_index=edge_index, edge_weight=edge_weight, pair_index=pair_index, pair_weight=pair_weight, anchor_ids=anchor_ids, offsets=offsets, pos_nodes=pos_nodes, pos_weights=pos_weights, direct_pos_neighbors=torch.tensor(direct_neighbors).long(), direct_pos_offsets=torch.tensor(direct_offsets).long(), hop_neighbors=hop_neighbors, hop_offsets=hop_offsets, degree=degree_tensor, num_nodes=x_text.size(0))

    print(f"[Graph] nodes={len(items_df)}, graph_edges={edge_index.size(1)}, loss_pairs={pair_index.size(1)}, text_dim={text_dim}, image_dim={img_dim}")
    print(f"[Degree] mean={degree.mean():.4f}, median={np.median(degree):.4f}, max={degree.max()}")
    return data, items_df

class Fusion(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, dropout=0.1, fusion_type="gate"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "gate":
            self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            self.img_proj = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        else:
            self.proj = nn.Sequential(nn.Linear(text_dim + image_dim + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x_text, x_image, image_mask):
        x_text = F.normalize(x_text, p=2, dim=-1)
        x_image = F.normalize(x_image, p=2, dim=-1)
        if self.fusion_type == "concat":
            return self.proj(torch.cat([x_text, x_image, image_mask], dim=-1))
        t, v = self.text_proj(x_text), self.img_proj(x_image)
        g = self.gate(torch.cat([t, v, image_mask], dim=-1))
        return image_mask * (g * t + (1.0 - g) * v) + (1.0 - image_mask) * t

class GraphEncoder(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=256, out_dim=128, dropout=0.1, fusion_type="gate"):
        super().__init__()
        self.fusion = Fusion(text_dim, image_dim, hidden_dim, dropout, fusion_type)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_with_graph(self, data, edge_index, edge_weight):
        x = self.fusion(data.x_text, data.x_image, data.image_mask)
        x = self.dropout(F.relu(self.conv1(x, edge_index, edge_weight)))
        h = self.conv2(x, edge_index, edge_weight)
        return h, F.normalize(h, p=2, dim=-1)

    def forward(self, data, return_pre_norm=False):
        h, z = self.encode_with_graph(data, data.edge_index, data.edge_weight)
        return (h, z) if return_pre_norm else z

class NegativeSampler:
    def __init__(self, num_nodes, pair_index, direct_pos_neighbors, direct_pos_offsets, hop_neighbors, hop_offsets, seed=42):
        self.num_nodes = int(num_nodes)
        self.src_np = pair_index[0].detach().cpu().numpy()
        self.dst_np = pair_index[1].detach().cpu().numpy()
        self.rng = np.random.default_rng(seed)
        d_n, d_o = direct_pos_neighbors.cpu().numpy(), direct_pos_offsets.cpu().numpy()
        h_n, h_o = hop_neighbors.cpu().numpy(), hop_offsets.cpu().numpy()
        self.forbid_sets, self.hard_pools = [], []
        for i in range(self.num_nodes):
            direct = set(d_n[int(d_o[i]):int(d_o[i + 1])].tolist())
            direct.add(i)
            hop = set(h_n[int(h_o[i]):int(h_o[i + 1])].tolist())
            self.forbid_sets.append(direct)
            self.hard_pools.append(np.array(sorted(list(hop - direct)), dtype=np.int64))

    def refresh_cache(self, num_neg, hard_neg, max_try=30, device=None):
        neg_idx = np.empty((len(self.src_np), num_neg), dtype=np.int64)
        hard_neg = min(hard_neg, num_neg)
        for i, (s, d) in enumerate(zip(self.src_np, self.dst_np)):
            s, d = int(s), int(d)
            forbid, selected = self.forbid_sets[s], []
            pool = self.hard_pools[s]
            if pool.size > 0 and hard_neg > 0:
                pool = pool[pool != d]
                if pool.size > 0:
                    selected += self.rng.choice(pool, size=min(hard_neg, pool.size), replace=False).tolist()
            tries = 0
            while len(selected) < num_neg and tries < max_try * num_neg:
                cand = int(self.rng.integers(0, self.num_nodes))
                if cand != d and cand not in forbid:
                    selected.append(cand)
                tries += 1
            while len(selected) < num_neg:
                cand = int(self.rng.integers(0, self.num_nodes))
                if cand != d and cand not in forbid:
                    selected.append(cand)
            neg_idx[i] = np.asarray(selected[:num_neg], dtype=np.int64)
        neg_idx = torch.from_numpy(neg_idx)
        return neg_idx.to(device, non_blocking=True) if device is not None else neg_idx

def pair_loss(z, pair_index, pair_weight, neg_idx, tau=0.1, chunk_size=512, link=False):
    src_all, dst_all = pair_index[0], pair_index[1]
    total_loss, total_weight = z.new_tensor(0.0), z.new_tensor(0.0)
    for st in range(0, src_all.numel(), chunk_size):
        ed = min(st + chunk_size, src_all.numel())
        src, dst, neg, w = src_all[st:ed], dst_all[st:ed], neg_idx[st:ed].to(z.device), pair_weight[st:ed]
        z_src, z_pos, z_neg = z[src], z[dst], z[neg]
        pos = (z_src * z_pos).sum(dim=-1) / tau
        neg = (z_src.unsqueeze(1) * z_neg).sum(dim=-1) / tau
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        if link:
            labels = torch.zeros_like(logits); labels[:, 0] = 1.0
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)
        else:
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=z.device)
            loss = F.cross_entropy(logits, labels, reduction="none")
        total_loss += (w * loss).sum()
        total_weight += w.sum()
    return total_loss / total_weight.clamp_min(1e-12)

def set_loss(z, anchor_ids, offsets, pos_nodes, pos_weights):
    losses = []
    for i, a in enumerate(anchor_ids):
        st, ed = offsets[i].item(), offsets[i + 1].item()
        ns, ws = pos_nodes[st:ed], pos_weights[st:ed]
        if ns.numel() == 0:
            continue
        proto = F.normalize(((ws.unsqueeze(-1) * z[ns]).sum(dim=0) / ws.sum().clamp_min(1e-12)).unsqueeze(0), p=2, dim=-1).squeeze(0)
        losses.append(1.0 - F.cosine_similarity(z[a].unsqueeze(0), proto.unsqueeze(0), dim=-1))
    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)

def var_loss(h, target=1.0):
    return torch.mean(F.relu(target - torch.sqrt(h.var(dim=0) + 1e-4)))

def cov_loss(h):
    h = h - h.mean(dim=0, keepdim=True)
    n, d = h.shape
    cov = (h.T @ h) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / (d * max(d - 1, 1))

def save_ckpt(path, model, data, args):
    torch.save({"model_state_dict": model.state_dict(), "config": {"text_dim": data.x_text.size(1), "image_dim": data.x_image.size(1), "hidden_dim": args.hidden_dim, "out_dim": args.out_dim, "dropout": args.dropout, "fusion_type": args.fusion_type}}, path)

def train(args):
    global LEVEL_WEIGHT, DEGREE_PENALTY_ETA, MAX_POS_PER_ANCHOR
    set_seed(args.seed)
    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() and args.gpu_num >= 0 else "cpu"
    LEVEL_WEIGHT = {"strong": args.strong_w, "weak": args.weak_w, "loose": args.loose_w, "unknown": args.unknown_w}
    DEGREE_PENALTY_ETA, MAX_POS_PER_ANCHOR = args.degree_penalty_eta, args.max_pos_per_anchor

    data, items_df = build_graph_data(args.items_path, args.pos_edges_path, args.multimodal_emb_path, parse_levels(args.use_graph_levels), parse_levels(args.use_loss_levels))
    sampler = NegativeSampler(data.num_nodes, data.pair_index, data.direct_pos_neighbors, data.direct_pos_offsets, data.hop_neighbors, data.hop_offsets, args.seed)
    neg_cache = sampler.refresh_cache(args.num_neg_per_pos, args.hard_neg_per_pos, args.neg_sample_try, device=device if args.pin_neg_cache_to_gpu else None)

    for k in ["x_text", "x_image", "image_mask", "edge_index", "edge_weight", "pair_index", "pair_weight", "anchor_ids", "offsets", "pos_nodes", "pos_weights", "degree"]:
        data[k] = data[k].to(device)

    model = GraphEncoder(data.x_text.size(1), data.x_image.size(1), args.hidden_dim, args.out_dim, args.dropout, args.fusion_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = os.path.join(args.save_dir, datetime.now().strftime("%y%m%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    best_loss, best_path, prev_best = float("inf"), None, None

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        if epoch == 1 or ((epoch - 1) % args.neg_cache_refresh_every == 0 and epoch != 1):
            neg_cache = sampler.refresh_cache(args.num_neg_per_pos, args.hard_neg_per_pos, args.neg_sample_try, device=device if args.pin_neg_cache_to_gpu else None)

        h, z = model.encode_with_graph(data, data.edge_index, data.edge_weight)
        l_pair = pair_loss(z, data.pair_index, data.pair_weight, neg_cache, args.contrastive_tau, args.chunk_size, link=False)
        l_link = pair_loss(z, data.pair_index, data.pair_weight, neg_cache, args.contrastive_tau, args.chunk_size, link=True)
        l_set = set_loss(z, data.anchor_ids, data.offsets, data.pos_nodes, data.pos_weights)
        l_var, l_cov = var_loss(h, args.var_target), cov_loss(h)
        loss = args.lambda_pair * l_pair + args.lambda_set * l_set + args.lambda_link * l_link + args.lambda_var * l_var + args.lambda_cov * l_cov
        loss.backward(); optimizer.step()

        if epoch % args.best_save_every == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            if prev_best and os.path.exists(prev_best):
                os.remove(prev_best)
            best_path = os.path.join(out_dir, f"best_graph_encoder_{epoch}.pt")
            save_ckpt(best_path, model, data, args)
            prev_best = best_path

        with torch.no_grad():
            sample_n = min(20000, data.pair_index.size(1))
            idx = torch.randperm(data.pair_index.size(1), device=device)[:sample_n]
            pair_cos = F.cosine_similarity(z[data.pair_index[0, idx]], z[data.pair_index[1, idx]], dim=-1).mean().item()
            h_norm = h.norm(dim=-1).mean().item()

        print(f"[Epoch {epoch:03d}] loss={loss.item():.6f} pair={l_pair.item():.6f} set={l_set.item():.6f} link={l_link.item():.6f} var={l_var.item():.6f} cov={l_cov.item():.6f} h_norm={h_norm:.6f} pair_cos={pair_cos:.6f}")

    last_path = os.path.join(out_dir, "last_graph_encoder.pt")
    save_ckpt(last_path, model, data, args)
    best_path = best_path or last_path
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    model.eval()

    with torch.no_grad():
        _, z = model(data, return_pre_norm=True)
        z = z.cpu()

    torch.save({"embeddings": z, "asins": items_df["asin"].tolist(), "new_item_ids": items_df["new_item_id"].tolist(), "dim": z.size(1), "fusion_type": args.fusion_type}, os.path.join(out_dir, "item_graph_embeddings.pt"))
    print(f"Best model saved to: {best_path}")
    print(f"Saved embeddings to: {os.path.join(out_dir, 'item_graph_embeddings.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--pos_edges_path", type=str, required=True)
    parser.add_argument("--multimodal_emb_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/graph_item")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion_type", type=str, default="gate", choices=["gate", "concat"])
    parser.add_argument("--use_graph_levels", type=str, default="strong,weak,loose")
    parser.add_argument("--use_loss_levels", type=str, default="strong,weak,loose")
    parser.add_argument("--strong_w", type=float, default=1.0)
    parser.add_argument("--weak_w", type=float, default=0.6)
    parser.add_argument("--loose_w", type=float, default=0.2)
    parser.add_argument("--unknown_w", type=float, default=0.4)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    parser.add_argument("--lambda_cov", type=float, default=0.01)
    parser.add_argument("--lambda_link", type=float, default=0.2)
    parser.add_argument("--lambda_pair", type=float, default=1.0)
    parser.add_argument("--lambda_set", type=float, default=0.1)
    parser.add_argument("--var_target", type=float, default=1.0)
    parser.add_argument("--num_neg_per_pos", type=int, default=8)
    parser.add_argument("--hard_neg_per_pos", type=int, default=3)
    parser.add_argument("--contrastive_tau", type=float, default=0.1)
    parser.add_argument("--neg_sample_try", type=int, default=30)
    parser.add_argument("--degree_penalty_eta", type=float, default=0.2)
    parser.add_argument("--max_pos_per_anchor", type=int, default=20)
    parser.add_argument("--neg_cache_refresh_every", type=int, default=5)
    parser.add_argument("--best_save_every", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--pin_neg_cache_to_gpu", action="store_true")
    args = parser.parse_args()
    train(args)