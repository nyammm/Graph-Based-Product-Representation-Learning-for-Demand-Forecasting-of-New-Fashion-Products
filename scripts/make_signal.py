import os, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def ensure_numpy_1d(x):
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.asarray(x, dtype=np.float32).reshape(-1)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_item_emb(path):
    obj = load_pkl(path)
    return {str(k): ensure_numpy_1d(v) for k, v in obj.items()}

def get_sales_cols(horizon):
    return [str(i) for i in range(horizon)]

def l2_normalize(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def make_time_features(date_matrix):
    dates = pd.to_datetime(date_matrix.reshape(-1))
    week = dates.isocalendar().week.to_numpy(dtype=np.float32)
    month = dates.month.to_numpy(dtype=np.float32)
    feats = np.stack([np.sin(2 * np.pi * week / 52.0), np.cos(2 * np.pi * week / 52.0), np.sin(2 * np.pi * month / 12.0), np.cos(2 * np.pi * month / 12.0)], axis=-1).astype(np.float32)
    return feats.reshape(*date_matrix.shape, 4)

def prepare_train_pool(train_df, emb_dict, id_col, date_col, horizon):
    df = train_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=False)
    ids = df[id_col].astype(str).to_numpy()
    dates = df[date_col].to_numpy(dtype="datetime64[ns]")
    sales = df[get_sales_cols(horizon)].to_numpy(dtype=np.float32)
    emb = l2_normalize(np.stack([emb_dict[x] for x in ids]).astype(np.float32), axis=1)
    return {"ids": ids, "dates": dates, "sales": sales, "emb": emb, "orig_idx": df["index"].to_numpy()}

def build_neighbor_tensors(target_df, train_pool, emb_dict, id_col, date_col, horizon, top_k, sim_threshold):
    df = target_df.copy().reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    train_ids, train_dates, train_sales, train_emb, train_orig_idx = train_pool["ids"], train_pool["dates"], train_pool["sales"], train_pool["emb"], train_pool["orig_idx"]
    target_ids = df[id_col].astype(str).to_numpy()
    target_dates = df[date_col].to_numpy(dtype="datetime64[ns]")
    N, K, H = len(df), top_k, horizon
    neighbor_sales = np.zeros((N, K, H), dtype=np.float32)
    neighbor_mask = np.zeros((N, K, H), dtype=np.float32)
    neighbor_time_feats = np.zeros((N, K, H, 4), dtype=np.float32)
    neighbor_sims = np.zeros((N, K), dtype=np.float32)
    neighbor_valid = np.zeros((N, K), dtype=np.float32)
    neighbor_ids = [["" for _ in range(K)] for _ in range(N)]
    neighbor_train_indices = [[-1 for _ in range(K)] for _ in range(N)]
    week_idx = np.arange(H)[None, :]

    for i, (tid, tdate) in enumerate(zip(target_ids, target_dates)):
        target_emb = emb_dict[str(tid)]
        target_emb = target_emb / (np.linalg.norm(target_emb) + 1e-12)
        cut = np.searchsorted(train_dates, tdate, side="left")
        if cut == 0:
            continue
        sims = train_emb[:cut] @ target_emb
        cand_idx = np.arange(cut)
        if sim_threshold is not None:
            keep = sims >= sim_threshold
            if not np.any(keep):
                continue
            cand_idx = np.flatnonzero(keep)
            sims_kept = sims[keep]
        else:
            sims_kept = sims
        k = min(K, len(sims_kept))
        if k == 0:
            continue
        part = np.argpartition(-sims_kept, kth=k - 1)[:k]
        top_local = part[np.argsort(-sims_kept[part])]
        top_idx = cand_idx[top_local]
        selected_ids = train_ids[top_idx]
        selected_dates = train_dates[top_idx]
        selected_sales = train_sales[top_idx].astype(np.float32)
        gap_weeks = ((tdate - selected_dates).astype("timedelta64[D]").astype(np.int64) // 7)
        available_len = np.clip(gap_weeks, 0, H).astype(np.int32)
        valid_mask = (week_idx < available_len[:, None]) & np.isfinite(selected_sales)
        neighbor_week_dates = selected_dates[:, None].astype("datetime64[D]") + (np.arange(H)[None, :] * 7).astype("timedelta64[D]")
        time_feats = make_time_features(neighbor_week_dates)

        neighbor_sales[i, :k, :] = np.where(valid_mask, selected_sales, 0.0)
        neighbor_mask[i, :k, :] = valid_mask.astype(np.float32)
        neighbor_time_feats[i, :k, :, :] = np.where(valid_mask[:, :, None], time_feats, 0.0).astype(np.float32)
        neighbor_sims[i, :k] = sims[top_idx].astype(np.float32)
        neighbor_valid[i, :k] = (valid_mask.sum(axis=1) > 0).astype(np.float32)

        for j in range(k):
            neighbor_ids[i][j] = str(selected_ids[j])
            neighbor_train_indices[i][j] = int(train_orig_idx[top_idx[j]])

    return {
        "target_ids": [str(x) for x in target_ids.tolist()],
        "neighbor_ids": neighbor_ids,
        "neighbor_train_indices": neighbor_train_indices,
        "neighbor_sales": torch.from_numpy(neighbor_sales),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
        "neighbor_time_feats": torch.from_numpy(neighbor_time_feats),
        "neighbor_sims": torch.from_numpy(neighbor_sims),
        "neighbor_valid": torch.from_numpy(neighbor_valid),
    }

def main(args):
    df = pd.read_csv(args.data_path)
    emb_dict = load_item_emb(args.item_emb_path)
    train_list = load_pkl(args.train_list_path)
    valid_list = load_pkl(args.valid_list_path)
    test_list = load_pkl(args.test_list_path)

    train_df = df[df[args.id_col].astype(str).isin(set(map(str, train_list)))].copy()
    valid_df = df[df[args.id_col].astype(str).isin(set(map(str, valid_list)))].copy()
    test_df = df[df[args.id_col].astype(str).isin(set(map(str, test_list)))].copy()

    train_pool = prepare_train_pool(train_df, emb_dict, args.id_col, args.date_col, args.horizon)
    sim_threshold = None if args.sim_threshold < 0 else args.sim_threshold

    train_out = build_neighbor_tensors(train_df, train_pool, emb_dict, args.id_col, args.date_col, args.horizon, args.top_k, sim_threshold)
    valid_out = build_neighbor_tensors(valid_df, train_pool, emb_dict, args.id_col, args.date_col, args.horizon, args.top_k, sim_threshold)
    test_out = build_neighbor_tensors(test_df, train_pool, emb_dict, args.id_col, args.date_col, args.horizon, args.top_k, sim_threshold)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_out, save_dir / "train_signal.pt")
    torch.save(valid_out, save_dir / "valid_signal.pt")
    torch.save(test_out, save_dir / "test_signal.pt")

    print("Saved:")
    print(save_dir / "train_signal.pt")
    print(save_dir / "valid_signal.pt")
    print(save_dir / "test_signal.pt")
    print("train neighbor_sales:", train_out["neighbor_sales"].shape)
    print("valid neighbor_sales:", valid_out["neighbor_sales"].shape)
    print("test neighbor_sales:", test_out["neighbor_sales"].shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_list_path", type=str, required=True)
    parser.add_argument("--valid_list_path", type=str, required=True)
    parser.add_argument("--test_list_path", type=str, required=True)
    parser.add_argument("--item_emb_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./data/neighbor")
    parser.add_argument("--id_col", type=str, default="item_number_color")
    parser.add_argument("--date_col", type=str, default="release_date")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sim_threshold", type=float, default=0.6)
    args = parser.parse_args()
    main(args)