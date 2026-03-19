import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class TupleTensorDataset(Dataset):
    """Simple dataset for (x_lin, x_nonpar, y) tuples."""

    def __init__(self, x_lin, x_nonpar, y):
        self.x_lin = x_lin
        self.x_nonpar = x_nonpar
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_lin[idx], self.x_nonpar[idx], self.y[idx]


class TensorDataset(Dataset):
    """Dataset for single input tensor and target."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class DPLLightning(pl.LightningModule):
    def __init__(self, net, lr, weight_decay=0.0):
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters(ignore=["net"])

    def forward(self, x_lin, x_nonpar):
        return self.net(x_lin, x_nonpar)

    def training_step(self, batch, batch_idx):
        x_lin, x_nonpar, y = batch
        preds = self(x_lin, x_nonpar)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x_lin, x_nonpar, y = batch
        preds = self(x_lin, x_nonpar)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class MNetLightning(pl.LightningModule):
    def __init__(self, net, lr, weight_decay=0.0):
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters(ignore=["net"])

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # total_steps = self.trainer.estimated_stepping_batches
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     total_steps=total_steps,
        #     pct_start=0.2,
        #     anneal_strategy="cos",
        #     div_factor=25,
        #     final_div_factor=1000,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #     },
        # }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.6,
            patience=3,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
        



def predict_net_dpl(net, x_lin, x_nonpar, batch_size):
    """Batch predictions for DPLNetSparse."""
    ds = TupleTensorDataset(x_lin, x_nonpar, torch.zeros(len(x_lin)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    net.eval()
    with torch.no_grad():
        for xb_lin, xb_nonpar, _ in loader:
            preds.append(net(xb_lin, xb_nonpar))
    return torch.cat(preds, dim=0)


def predict_net_m(net, x, batch_size):
    """Batch predictions for mNet."""
    ds = TensorDataset(x, torch.zeros(len(x)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    net.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(net(xb))
    return torch.cat(preds, dim=0)



class TrainLossEarlyStop(pl.Callback):
    def __init__(self, monitor="train_loss", mode="min", patience=5, min_delta=0.0, verbose=True):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.verbose = verbose

        self.best = None
        self.num_bad_epochs = 0
        self.best_epoch = None
        self.best_state = None  # keep best weights in memory

    def _improved(self, cur, best):
        if self.mode == "min":
            return cur < (best - self.min_delta)
        else:
            return cur > (best + self.min_delta)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        cur = trainer.callback_metrics.get(self.monitor, None)
        if cur is None:
            return  # train_loss not logged (see training_step note below)

        cur = float(cur.detach().cpu())

        if self.best is None or self._improved(cur, self.best):
            self.best = cur
            self.best_epoch = trainer.current_epoch
            self.num_bad_epochs = 0

            # store best weights on CPU
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in pl_module.state_dict().items()}

            if self.verbose and trainer.is_global_zero:
                print(f"[train best] epoch={self.best_epoch} {self.monitor}={self.best:.6f}", flush=True)
        else:
            self.num_bad_epochs += 1
            if self.verbose and trainer.is_global_zero:
                print(f"[train no-improve] epoch={trainer.current_epoch} {self.monitor}={cur:.6f} "
                      f"({self.num_bad_epochs}/{self.patience})", flush=True)

            if self.num_bad_epochs >= self.patience:
                if self.verbose and trainer.is_global_zero:
                    print("[stop] training loss converged (patience reached).", flush=True)
                trainer.should_stop = True  # stop training

    def on_fit_end(self, trainer, pl_module):
        # restore best weights (optional but usually desired)
        if self.best_state is not None:
            pl_module.load_state_dict(self.best_state)
            if self.verbose and trainer.is_global_zero:
                print(f"[restore] best train weights from epoch={self.best_epoch} ({self.monitor}={self.best:.6f})", flush=True)


class PrintTrainValLossEveryN(pl.Callback):
    def __init__(self, every_n_steps=50, train_key="train_loss", val_key="val_loss"):
        self.every_n_steps = every_n_steps
        self.train_key = train_key
        self.val_key = val_key

    def _get_metric_float(self, trainer, key):
        v = trainer.callback_metrics.get(key)
        if v is None:
            return None
        # callback_metrics values are usually tensors
        try:
            return float(v.detach().cpu())
        except Exception:
            try:
                return float(v)
            except Exception:
                return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps != 0:
            return

        train_loss = self._get_metric_float(trainer, self.train_key)
        val_loss = self._get_metric_float(trainer, self.val_key)

        # fallback to outputs for train loss if not logged
        if train_loss is None:
            loss = None
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif hasattr(outputs, "detach"):
                loss = outputs
            if loss is not None:
                train_loss = float(loss.detach().cpu())

        if train_loss is None and val_loss is None:
            return

        msg = f"step={trainer.global_step}"
        if train_loss is not None:
            msg += f" train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.6f}"  # latest available val loss
        else:
            msg += " val_loss=NA"  # not computed yet (no val run has happened)

        pl_module.print(msg, flush=True)
