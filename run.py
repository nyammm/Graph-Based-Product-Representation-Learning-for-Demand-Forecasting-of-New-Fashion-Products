import os
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm

from utils.load_data import DataModule
from utils.metric import get_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def forward_batch(model, batch, model_type):
    if model_type == "gtm":
        item_sales, temporal_features, gtrends, image_embs, text_embs = batch
        pred = model(temporal_features, gtrends, image_embs, text_embs)
    else:
        item_sales, temporal_features, gtrends, items, image_embs, text_embs, neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid = batch
        pred = model(temporal_features, gtrends, items, image_embs, text_embs, neighbor_sales, neighbor_mask, neighbor_time_feats, neighbor_sims, neighbor_valid)

    if isinstance(pred, dict):
        pred = pred.get("forecast", pred.get("pred", pred.get("preds", pred.get("y_hat", pred))))
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    return item_sales, pred


@torch.no_grad()
def inference(model, test_loader, model_type, device, scale_value, output_dim):
    gt, forecasts = [], []
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, total=len(test_loader), ascii=True):
        batch = [x.to(device) for x in batch]
        item_sales, y_pred = forward_batch(model, batch, model_type)
        gt.append(item_sales.detach().cpu().numpy()[:, :output_dim])
        forecasts.append(y_pred.detach().cpu().numpy()[:, :output_dim])

    gt = np.concatenate(gt, axis=0)
    forecasts = np.concatenate(forecasts, axis=0)
    rescaled_gt = gt * scale_value
    rescaled_forecasts = forecasts * scale_value
    
    mae, wape, adj_smape, accum_smape = get_score(rescaled_gt, rescaled_forecasts)
    
    print(f"MAE: {mae:.4f}")
    print(f"WAPE: {wape:.4f}")
    print(f"Adj-SMAPE: {adj_smape:.4f}")
    print(f"Accum-SMAPE: {accum_smape:.4f}")
    return gt, forecasts, rescaled_gt, rescaled_forecasts, {"mae": mae, "wape": wape, "adj_smape": adj_smape, "accum_smape": accum_smape}


def run(args):
    args.model_type = args.model_type.lower()
    args.data_dir = args.data_folder
    args.horizon = args.output_dim
    print(args, "\n")
    pl.seed_everything(args.seed, workers=True)

    dm = DataModule(args)
    dm.prepare_data()
    args.scale_value = dm.scale_value

    if args.model_type == "gtm":
        from models.GTM import GTM
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num,
            rescale_val=args.scale_value,
        )
    if args.model_type in ["ours", "graph"]:
        from models.Graph_transformer import GraphForecast
        model = GraphForecast(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num,
            rescale_val=args.scale_value,
            use_graph_item=args.use_graph_item,
            use_img=args.use_image,
            use_text=args.use_text,
            use_signal=args.use_signal,
        )

    string_dt = datetime.now().strftime("%y%m%d_%H%M")
    string_date = string_dt.split("_")[0]
    string_time = string_dt.split("_")[-1]
    run_name = f"{args.model_type}_{args.wandb_run}_seed{args.seed}"
    ckpt_dir = f"{args.log_dir}/{args.model_type}/{string_date}_{args.wandb_run}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{string_time}_{run_name}" + "---{epoch}---",
        monitor="val_adj_smape",
        mode="min",
        save_top_k=1,
    )

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=f"{run_name}_{string_dt}")
    logger = pl_loggers.WandbLogger()
    logger.watch(model)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    if args.mode == "train":
        trainer.fit(model, datamodule=dm)
        ckpt_path = checkpoint_callback.best_model_path
    else:
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required when mode=test")
        dm.setup("test")
        ckpt_path = args.ckpt_path

    print("Best ckpt:", ckpt_path)
    model = load_ckpt(model, ckpt_path)

    if not hasattr(dm, "test_dataset"):
        dm.setup("test")

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    gt, forecasts, rescaled_gt, rescaled_forecasts, metrics = inference(model, dm.test_dataloader(), args.model_type, device, dm.scale_value, args.output_dim)

    item_codes = dm.test_df["id"].astype(str).tolist()
    result_dir = f"{args.result_path}/{string_date}_{args.wandb_run}"
    os.makedirs(result_dir, exist_ok=True)
    result_name = f"{Path(ckpt_path).stem}_{args.output_dim}.pth"
    torch.save(
        {"results": rescaled_forecasts, "gts": rescaled_gt, "raw_results": forecasts, "raw_gts": gt, "codes": item_codes, "metrics": metrics, "scale_value": dm.scale_value, "ckpt_path": ckpt_path},
        Path(result_dir) / result_name,
    )
    print("Result saved:", Path(result_dir) / result_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot sales forecasting")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--result_path", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="visuelle")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--model_type", type=str, default="ours", choices=["gtm", "ours", "graph"])
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--use_graph_item", type=bool, default=True)
    parser.add_argument("--use_text", type=bool, default=True)
    parser.add_argument("--use_image", type=bool, default=True)
    parser.add_argument("--use_signal", type=bool, default=True)

    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_proj", type=str, default="")
    parser.add_argument("--wandb_run", type=str, default="")

    args = parser.parse_args()
    run(args)