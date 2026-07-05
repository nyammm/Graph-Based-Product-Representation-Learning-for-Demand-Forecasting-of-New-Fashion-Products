import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers.optimization import Adafactor


class BaseForecastModel(pl.LightningModule):
    def __init__(self, rescale_value=1820):
        super().__init__()
        self.rescale_value = rescale_value

    def _predict_from_batch(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        item_sales = train_batch[0]
        forecasted_sales = self._predict_from_batch(train_batch)
        loss = F.mse_loss(item_sales, forecasted_sales)
        log_dict, _ = self._calc_metrics(item_sales, forecasted_sales, "train")
        log_dict["train_loss"] = loss
        self.log_dict(log_dict, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales = val_batch[0]
        forecasted_sales = self._predict_from_batch(val_batch)
        return {"y": item_sales, "pred": forecasted_sales}

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.cat([x["y"] for x in val_step_outputs], dim=0)
        forecasted_sales = torch.cat([x["pred"] for x in val_step_outputs], dim=0)
        log_dict, smape_t = self._calc_metrics(item_sales, forecasted_sales, "val")
        self.log_dict(log_dict, logger=True, on_epoch=True)
        for t in range(smape_t.shape[1]):
            self.log(f"val_smape_w{t+1}", smape_t[:, t].mean(), on_epoch=True, prog_bar=False, logger=True)

    def _calc_metrics(self, item_sales, forecasted_sales, prefix):
        rescaled_item_sales = item_sales * self.rescale_value
        rescaled_forecasted_sales = forecasted_sales * self.rescale_value
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        eps = 1e-8
        trend = rescaled_item_sales
        outputs = rescaled_forecasted_sales
        wape = torch.sum(torch.abs(trend - outputs)) / (torch.sum(torch.abs(trend)) + eps)
        smape_t = torch.abs(trend - outputs) / (torch.abs(trend) + torch.abs(outputs) + eps)
        adj_smape_mean = smape_t.mean(dim=-1).mean()
        sum_trend = trend.sum(dim=-1)
        sum_out = outputs.sum(dim=-1)
        accum_smape_mean = (torch.abs(sum_trend - sum_out) / (torch.abs(sum_trend) + torch.abs(sum_out) + eps)).mean()
        return {f"{prefix}_loss": loss, f"{prefix}_mae": mae, f"{prefix}_adj_smape": adj_smape_mean, f"{prefix}_accum_smape": accum_smape_mean, f"{prefix}_wape": wape}, smape_t
