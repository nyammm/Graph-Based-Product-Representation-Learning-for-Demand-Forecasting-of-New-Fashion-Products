import os, json, math, argparse
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


def load_jsonl_asin_outer(path: str) -> Dict[str, Dict[str, Any]]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and len(obj) == 1:
                asin, info = next(iter(obj.items()))
                if isinstance(info, dict):
                    data[str(asin)] = info
    return data

def safe_log1p(value: Optional[float]) -> float:
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if pd.isna(value) or value < 0 else math.log1p(value)

def compute_log_gap(v1, v2) -> float:
    lv1, lv2 = safe_log1p(v1), safe_log1p(v2)
    return np.nan if pd.isna(lv1) or pd.isna(lv2) else abs(lv1 - lv2)

def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))

def limit_max_degree(edges_df: pd.DataFrame, max_degree: int, weight_col: str = "pos_weight") -> pd.DataFrame:
    if edges_df.empty or max_degree <= 0:
        return edges_df.iloc[0:0].copy()
    degree_count, selected = {}, []
    for idx, row in edges_df.sort_values(weight_col, ascending=False, na_position="last").iterrows():
        s, d = row["src_asin"], row["dst_asin"]
        if s == d:
            continue
        if degree_count.get(s, 0) >= max_degree or degree_count.get(d, 0) >= max_degree:
            continue
        selected.append(idx)
        degree_count[s] = degree_count.get(s, 0) + 1
        degree_count[d] = degree_count.get(d, 0) + 1
    return edges_df.loc[selected].reset_index(drop=True)

def build_items_df(raw_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item_id, (asin, info) in enumerate(raw_data.items()):
        price, rank = info.get("price"), info.get("rank")
        rows.append({
            "item_id": item_id,
            "asin": asin,
            "title": info.get("title"),
            "price": price,
            "rank": rank,
            "image_url": info.get("image"),
            "price_log": safe_log1p(price),
            "rank_log": safe_log1p(rank),
            "price_missing": int(price is None or pd.isna(price)),
            "rank_missing": int(rank is None or pd.isna(rank)),
        })
    return pd.DataFrame(rows)

def build_edges_df(raw_data, items_df, strong_thr, weak_thr, max_degree):
    item_lookup = items_df.set_index("asin").to_dict("index")
    asin_set = set(raw_data.keys())
    rows = []

    for src_asin, src_info in raw_data.items():
        src_item = item_lookup[src_asin]
        for dst_asin in set(src_info.get("also_buy", []) or []):
            if dst_asin not in asin_set or src_asin == dst_asin:
                continue
            dst_item = item_lookup[dst_asin]
            rank_gap = compute_log_gap(src_item["rank"], dst_item["rank"])
            price_gap = compute_log_gap(src_item["price"], dst_item["price"])

            if pd.isna(rank_gap):
                pos_level = "unknown"
            elif rank_gap < strong_thr:
                pos_level = "strong"
            elif rank_gap < weak_thr:
                pos_level = "weak"
            else:
                pos_level = "loose"

            if pd.isna(rank_gap):
                pos_weight = 0.5 if pd.isna(price_gap) else math.exp(-0.3 * price_gap)
            else:
                pos_weight = math.exp(-1.0 * rank_gap) if pd.isna(price_gap) else math.exp(-1.0 * rank_gap) * math.exp(-0.3 * price_gap)

            rows.append({
                "src_asin": src_asin,
                "dst_asin": dst_asin,
                "src_item_id": src_item["item_id"],
                "dst_item_id": dst_item["item_id"],
                "src_title": src_item["title"],
                "dst_title": dst_item["title"],
                "src_price": src_item["price"],
                "dst_price": dst_item["price"],
                "src_rank": src_item["rank"],
                "dst_rank": dst_item["rank"],
                "rank_gap": rank_gap,
                "price_gap": price_gap,
                "rank_missing_pair": int(pd.isna(rank_gap)),
                "price_missing_pair": int(pd.isna(price_gap)),
                "pos_level": pos_level,
                "pos_weight": pos_weight,
            })

    edges_df = pd.DataFrame(rows)
    if edges_df.empty:
        return edges_df

    edges_df["pair_key"] = edges_df.apply(lambda r: canonical_pair(r["src_asin"], r["dst_asin"]), axis=1)
    edges_df = edges_df.sort_values("pos_weight", ascending=False, na_position="last").drop_duplicates("pair_key").drop(columns=["pair_key"]).reset_index(drop=True)
    return limit_max_degree(edges_df, max_degree=max_degree, weight_col="pos_weight")

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    raw_data = load_jsonl_asin_outer(args.input_jsonl)
    items_df = build_items_df(raw_data)
    edges_df = build_edges_df(raw_data, items_df, args.strong_rank_gap_thr, args.weak_rank_gap_thr, args.max_degree)
    pos_edges_df = edges_df.copy()

    items_path = os.path.join(args.save_dir, "items.parquet")
    edges_path = os.path.join(args.save_dir, "edges.parquet")
    pos_edges_path = os.path.join(args.save_dir, "pos_edges.parquet")

    items_df.to_parquet(items_path, index=False)
    edges_df.to_parquet(edges_path, index=False)
    pos_edges_df.to_parquet(pos_edges_path, index=False)

    print(f"items_df: {items_df.shape}")
    print(f"edges_df: {edges_df.shape}")
    if not edges_df.empty:
        deg = pd.concat([edges_df["src_asin"], edges_df["dst_asin"]], ignore_index=True).value_counts()
        print(f"degree max={deg.max()}, mean={deg.mean():.4f}, median={deg.median():.4f}")
    print(f"saved: {items_path}")
    print(f"saved: {edges_path}")
    print(f"saved: {pos_edges_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/graph")
    parser.add_argument("--strong_rank_gap_thr", type=float, default=0.75)
    parser.add_argument("--weak_rank_gap_thr", type=float, default=2.5)
    parser.add_argument("--max_degree", type=int, default=10)
    args = parser.parse_args()
    main(args)
