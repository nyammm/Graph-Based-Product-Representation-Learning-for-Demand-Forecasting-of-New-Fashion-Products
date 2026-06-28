import os
import torch
import pickle
import json
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.model_type = args.model_type.lower()
        self.dataset = args.dataset
        self.num_workers = getattr(args, "num_workers", 2)

        if self.model_type not in ["gtm", "ours", "graph"]:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_pkl(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(self.data_dir, "total_data.csv"))

        self.train_list = self._load_pkl(os.path.join(self.data_dir, "train_list.pkl"))
        self.valid_list = self._load_pkl(os.path.join(self.data_dir, "valid_list.pkl"))
        self.test_list = self._load_pkl(os.path.join(self.data_dir, "test_list.pkl"))

        self.train_df = self.df[self.df["id"].isin(self.train_list)].reset_index(drop=True)
        self.valid_df = self.df[self.df["id"].isin(self.valid_list)].reset_index(drop=True)
        self.test_df = self.df[self.df["id"].isin(self.test_list)].reset_index(drop=True)

        self.cat_trend = self._load_pkl(os.path.join(self.data_dir, "total_cat_trend.pkl"))
        self.col_trend = self._load_pkl(os.path.join(self.data_dir, "total_col_trend.pkl"))
        self.fab_trend = self._load_pkl(os.path.join(self.data_dir, "total_fab_trend.pkl"))

        self.text_emb_dict = self._load_pkl(os.path.join(self.data_dir, "fclip_text_emb.pkl"))
        self.img_emb_dict = self._load_pkl(os.path.join(self.data_dir, "fclip_img_emb.pkl"))

        if self.model_type in ["ours", "graph"]:
            self.item_emb_dict = self._load_pkl(os.path.join(self.data_dir, "graph_item_emb.pkl"))
            self.train_signal = torch.load(os.path.join(self.data_dir, "train_signal.pt"), map_location="cpu")
            self.valid_signal = torch.load(os.path.join(self.data_dir, "valid_signal.pt"), map_location="cpu")
            self.test_signal = torch.load(os.path.join(self.data_dir, "test_signal.pt"), map_location="cpu")
        
        if self.dataset == 'visuelle':
            self.scale_value = 1065
        elif self.dataset == 'tbh':
            self.scale_value = 1820
        else:
            self.scale_value = self.train_df[[str(i) for i in range(12)]].values.astype(np.float32).max()

    def _make_basic_dataset(self, data_df):
        dataset = BasicDataset(
            data_df=data_df,
            cat_trend=self.cat_trend,
            col_trend=self.col_trend,
            fab_trend=self.fab_trend,
            text_emb_dict=self.text_emb_dict,
            img_emb_dict=self.img_emb_dict,
            scale_value=self.scale_value,
        )
        return dataset.preprocess_data()

    def _make_expanded_dataset(self, data_df, signal_pack):
        dataset = ExpandedDataset(
            data_df=data_df,
            cat_trend=self.cat_trend,
            col_trend=self.col_trend,
            fab_trend=self.fab_trend,
            item_emb_dict=self.item_emb_dict,
            text_emb_dict=self.text_emb_dict,
            img_emb_dict=self.img_emb_dict,
            signal_pack=signal_pack,
            scale_value=self.scale_value,
        )
        return dataset.preprocess_data()

    def _make_dataset(self, data_df, signal_pack=None):
        if self.model_type == "gtm":
            return self._make_basic_dataset(data_df)
        if self.model_type in ["ours", "graph"]:
            return self._make_expanded_dataset(data_df, signal_pack)

    def setup(self, stage=None):
        if stage in ["fit", None]:
            if self.model_type == "gtm":
                self.train_dataset = self._make_dataset(self.train_df)
                self.valid_dataset = self._make_dataset(self.valid_df)
            else:
                self.train_dataset = self._make_dataset(self.train_df, self.train_signal)
                self.valid_dataset = self._make_dataset(self.valid_df, self.valid_signal)

        if stage in ["test", None]:
            if self.model_type == "gtm":
                self.test_dataset = self._make_dataset(self.test_df)
            else:
                self.test_dataset = self._make_dataset(self.test_df, self.test_signal)

    def _make_loader(self, dataset, train=False):
        return DataLoader(dataset, batch_size=self.batch_size if train else 1, shuffle=train, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=(self.num_workers > 0))

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._make_loader(self.valid_dataset, train=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, train=False)


class BasicDataset:
    def __init__(self, data_df, cat_trend, col_trend, fab_trend, text_emb_dict, img_emb_dict, scale_value=1820, img_dim=512, text_dim=512):
        
        self.data_df = data_df.reset_index(drop=True).copy()

        self.text_emb_dict = text_emb_dict
        self.img_emb_dict = img_emb_dict

        self.cat_trend = cat_trend
        self.col_trend = col_trend
        self.fab_trend = fab_trend

        self.scale_value = scale_value
        self.img_dim = img_dim
        self.text_dim = text_dim

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def _minmax(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        return MinMaxScaler().fit_transform(x).reshape(-1).astype(np.float32)

    def preprocess_data(self):
        data = self.data_df.copy().reset_index(drop=True)
        keys = data["id"].astype(str).to_numpy()
        N = len(data)

        sales_cols = [str(i) for i in range(12)]
        item_sales = torch.FloatTensor(data[sales_cols].values.astype(np.float32)) / self.scale_value
        temporal_features = torch.FloatTensor(data[["day", "week", "month"]].values.astype(np.float32))

        gtrends_np = np.zeros((N, 3, 52), dtype=np.float32)
        for i, key in enumerate(tqdm(keys, total=N, ascii=True, desc="Building trends")):
            cat = np.asarray(self.cat_trend.get(key, np.zeros(52, dtype=np.float32)), dtype=np.float32)
            col = np.asarray(self.col_trend.get(key, np.zeros(52, dtype=np.float32)), dtype=np.float32)
            fab = np.asarray(self.fab_trend.get(key, np.zeros(52, dtype=np.float32)), dtype=np.float32)

            gtrends_np[i, 0] = self._minmax(cat)
            gtrends_np[i, 1] = self._minmax(col)
            gtrends_np[i, 2] = self._minmax(fab)

        gtrends = torch.from_numpy(gtrends_np)

        image_np = np.zeros((N, self.img_dim), dtype=np.float32)
        text_np = np.zeros((N, self.text_dim), dtype=np.float32)

        for i, key in enumerate(tqdm(keys, total=N, ascii=True, desc="Building embeddings")):
            
            if key in self.img_emb_dict:
                image_np[i] = np.asarray(self.img_emb_dict[key], dtype=np.float32).reshape(-1)
                
            if key in self.text_emb_dict:
                text_np[i] = np.asarray(self.text_emb_dict[key], dtype=np.float32).reshape(-1)

        image_embs = torch.from_numpy(image_np)
        text_embs = torch.from_numpy(text_np)

        return TensorDataset(item_sales, temporal_features, gtrends, image_embs, text_embs)

    def get_loader(self, batch_size, train=True, num_workers=2, pin_memory=True):
        print("Starting dataset creation process...")
        
        dataset = self.preprocess_data()
        data_loader = DataLoader(dataset, batch_size=batch_size if train else 1, shuffle=train, 
                                 num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
        print("Done.")
        
        return data_loader
    
    
class ExpandedDataset:
    def __init__(self, data_df, cat_trend, col_trend, fab_trend, item_emb_dict, text_emb_dict, img_emb_dict, 
                 signal_pack=None, horizon=12, top_k=10, scale_value=1065, item_dim=128, img_dim=512, text_dim=512):
        
        self.data_df = data_df.reset_index(drop=True).copy()

        self.item_emb_dict = item_emb_dict
        self.text_emb_dict = text_emb_dict
        self.img_emb_dict = img_emb_dict

        self.cat_trend = cat_trend
        self.col_trend = col_trend
        self.fab_trend = fab_trend

        self.signal_pack = signal_pack
        self.horizon = horizon
        self.top_k = top_k
        self.scale_value = scale_value
        self.item_dim = item_dim
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.signal_id_to_idx = {str(k): i for i, k in enumerate(self.signal_pack["target_ids"])} if self.signal_pack is not None else {}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def _minmax(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        return MinMaxScaler().fit_transform(x).reshape(-1).astype(np.float32)

    def preprocess_data(self):
        data = self.data_df.copy().reset_index(drop=True)
        keys = data["id"].astype(str).to_numpy()
        N, H, K = len(data), self.horizon, self.top_k

        sales_cols = [str(i) for i in range(H)]
        item_sales = torch.FloatTensor(data[sales_cols].values.astype(np.float32)) / self.scale_value
        temporal_features = torch.FloatTensor(data[["day", "week", "month"]].values.astype(np.float32))

        gtrends_np = np.zeros((N, 3, 52), dtype=np.float32)
        for i, key in enumerate(tqdm(keys, total=N, ascii=True, desc="Building trends")):
            cat = np.asarray(self.cat_trend.get(key, np.zeros(52, dtype=np.float32)), dtype=np.float32)
            col = np.asarray(self.col_trend.get(key, np.zeros(52, dtype=np.float32)), dtype=np.float32)
            fab = np.asarray(self.fab_trend.get(key, np.zeros(52, dtype=np.float32)), dtype=np.float32)

            gtrends_np[i, 0] = self._minmax(cat)
            gtrends_np[i, 1] = self._minmax(col)
            gtrends_np[i, 2] = self._minmax(fab)

        gtrends = torch.from_numpy(gtrends_np)

        items_np = np.zeros((N, self.item_dim), dtype=np.float32)
        image_np = np.zeros((N, self.img_dim), dtype=np.float32)
        text_np = np.zeros((N, self.text_dim), dtype=np.float32)

        for i, key in enumerate(tqdm(keys, total=N, ascii=True, desc="Building embeddings")):
            items_np[i] = np.asarray(self.item_emb_dict[key], dtype=np.float32).reshape(-1)
            
            if key in self.img_emb_dict:
                image_np[i] = np.asarray(self.img_emb_dict[key], dtype=np.float32).reshape(-1)
                
            if key in self.text_emb_dict:
                text_np[i] = np.asarray(self.text_emb_dict[key], dtype=np.float32).reshape(-1)

        items = torch.from_numpy(items_np)
        image_embs = torch.from_numpy(image_np)
        text_embs = torch.from_numpy(text_np)

        neighbor_sales = torch.zeros(N, K, H, dtype=torch.float32)
        neighbor_mask = torch.zeros(N, K, H, dtype=torch.float32)
        neighbor_time_feats = torch.zeros(N, K, H, 4, dtype=torch.float32)
        neighbor_sims = torch.zeros(N, K, dtype=torch.float32)
        neighbor_valid = torch.zeros(N, K, dtype=torch.float32)

        if self.signal_pack is not None:
            signal_indices = np.array([self.signal_id_to_idx.get(str(k), -1) for k in keys], dtype=np.int64)
            valid_rows_np = signal_indices >= 0
            print(f"Missing signal rows: {int((~valid_rows_np).sum())} / {N}")

            if valid_rows_np.any():
                row_idx = torch.from_numpy(np.flatnonzero(valid_rows_np)).long()
                sig_idx = torch.from_numpy(signal_indices[valid_rows_np]).long()

                ns = self.signal_pack["neighbor_sales"][sig_idx].detach().cpu().float()
                nm = self.signal_pack["neighbor_mask"][sig_idx].detach().cpu().float()
                nt = self.signal_pack["neighbor_time_feats"][sig_idx].detach().cpu().float()
                sim = self.signal_pack["neighbor_sims"][sig_idx].detach().cpu().float()
                nv = self.signal_pack["neighbor_valid"][sig_idx].detach().cpu().float()

                k_avail = min(K, ns.shape[1])
                neighbor_sales[row_idx, :k_avail] = ns[:, :k_avail]
                neighbor_mask[row_idx, :k_avail] = nm[:, :k_avail]
                neighbor_time_feats[row_idx, :k_avail] = nt[:, :k_avail]
                neighbor_sims[row_idx, :k_avail] = sim[:, :k_avail]
                neighbor_valid[row_idx, :k_avail] = nv[:, :k_avail]

        return TensorDataset(item_sales, temporal_features, gtrends, items, image_embs, text_embs, 
                             neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid)

    def get_loader(self, batch_size, train=True, num_workers=2, pin_memory=True):
        print("Starting dataset creation process...")
        
        dataset = self.preprocess_data()
        data_loader = DataLoader(dataset, batch_size=batch_size if train else 1, shuffle=train, 
                                 num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
        print("Done.")
        
        return data_loader
