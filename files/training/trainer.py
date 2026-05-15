"""
ConformerFlow — Trainer (Config-driven)
Reads resolved config dict → builds model → trains end-to-end.
"""

import os, json, logging, time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast

from model.model_factory import build_model, ConformerFlowModel
from training.losses     import ConformerFlowLoss
from model.flow_matching import SE3FlowMatcher

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# DDP HELPERS
# ─────────────────────────────────────────────────────────

def setup_ddp() -> tuple:
    """
    Initialise DDP process group from torchrun environment variables.
    Returns (local_rank, world_size, is_main).
    """
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    is_main = (local_rank == 0)
    return local_rank, world_size, is_main


def cleanup_ddp():
    """Destroy DDP process group cleanly."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────
# LR SCHEDULER
# ─────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, decay_steps, min_lr=1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps  = decay_steps
        self.min_lr       = min_lr
        self.base_lrs     = [g["lr"] for g in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * base

    def _get_lr(self):
        s = self._step
        if s < self.warmup_steps:
            return s / max(self.warmup_steps, 1)
        progress = min((s - self.warmup_steps) / max(self.decay_steps, 1), 1.0)
        import math
        cos_val  = 0.5 * (1 + math.cos(math.pi * progress))
        min_frac = self.min_lr / self.base_lrs[0]
        return max(min_frac, cos_val)

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────

def _set_seed(seed: int):
    """Set global random seeds for full reproducibility."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


class Trainer:
    """
    Config-driven trainer for ConformerFlow.
    Reads all hyperparameters from the YAML config dict.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        tcfg     = cfg["training"]

        # Reproducibility — set before any model/data initialisation
        seed = tcfg.get("seed", 42)
        _set_seed(seed)
        if True:  # always log seed
            logger.info(f"Global random seed: {seed}")
        lcfg     = cfg["loss"]
        logcfg   = cfg["logging"]

        # FIX 5: DDP setup — must happen before any CUDA allocation
        self.multi_gpu = tcfg.get("multi_gpu", False)
        if self.multi_gpu:
            self.local_rank, self.world_size, self.is_main = setup_ddp()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank  = 0
            self.world_size  = 1
            self.is_main     = True
            self.device      = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if self.is_main:
            logger.info(f"Training on: {self.device}  "
                        f"(world_size={self.world_size})")

        # Build model on the correct device
        self.model = build_model(cfg).to(self.device)

        # FIX 5c: Wrap in DDP after moving to device
        if self.multi_gpu and self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids    = [self.local_rank],
                output_device = self.local_rank,
                find_unused_parameters = False,
            )
            if self.is_main:
                logger.info(f"Model wrapped in DDP across {self.world_size} GPUs")

        # Parameter count (unwrap DDP to access .count_parameters)
        raw_model = self.model.module if self.multi_gpu else self.model
        params    = raw_model.count_parameters()
        if self.is_main:
            logger.info(f"Parameters: total={params['total']:,}  "
                        f"encoder={params['encoder']:,}  "
                        f"generative={params['generative']:,}")

        # Loss
        self.loss_fn = ConformerFlowLoss(
            lambda_flow      = lcfg["lambda_flow"],
            lambda_ensemble  = lcfg["lambda_ensemble"],
            lambda_kl        = lcfg["lambda_kl"],
            lambda_diversity = lcfg["lambda_diversity"],
            lambda_geometry  = lcfg["lambda_geometry"],
            lambda_chirality = lcfg.get("lambda_chirality", 0.1),
            free_bits        = cfg["ensemble_stats"].get("free_bits", 0.5),
            min_spread       = lcfg.get("min_spread", 0.5),
        )

        # Flow matcher (for loss computation)
        self.flow_matcher = SE3FlowMatcher(
            sigma_min=cfg["generative_model"].get("sigma_min", 0.01)
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = tcfg["learning_rate"],
            weight_decay = tcfg["weight_decay"],
            betas        = (0.9, 0.999),
        )

        # LR scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps = tcfg["warmup_steps"],
            decay_steps  = tcfg["lr_decay_steps"],
        )

        # FIX 4: Mixed precision — bf16 vs fp16 have very different requirements.
        # fp16: needs GradScaler (values underflow to zero without it)
        # bf16: does NOT need GradScaler (same exponent range as fp32, no underflow)
        #       bf16 is what A100/H100/B200 are optimised for — much faster than fp16
        use_bf16  = tcfg.get("use_bf16", False)
        use_cuda  = self.device.type == "cuda"
        self.amp_dtype  = torch.bfloat16 if use_bf16 else torch.float16
        self.use_amp    = use_cuda
        # GradScaler: enabled only for fp16 (not bf16, not CPU)
        self.scaler = GradScaler("cuda", enabled=(use_cuda and not use_bf16))

        # EMA — exponential moving average of weights for stable inference
        # decay=0.9999 is standard for flow matching / diffusion models
        ema_decay = tcfg.get("ema_decay", 0.9999)
        self.ema_decay  = ema_decay
        self.ema_params = {
            name: param.data.clone().float()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Tracking
        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.gen_type      = cfg["generative_model"]["type"]

        # Dirs
        self.checkpoint_dir = Path(tcfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        self.use_wandb = logcfg.get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb()

        # Resume
        resume = tcfg.get("resume_from")
        if resume and Path(resume).exists():
            self._load_checkpoint(resume)

    def _init_wandb(self):
        try:
            import wandb
            logcfg = self.cfg["logging"]
            wandb.init(
                project = logcfg.get("project_name", "conformerflow"),
                name    = logcfg.get("run_name", "run"),
                config  = self.cfg,
            )
            self.wandb = wandb
        except ImportError:
            logger.warning("wandb not installed — console logging only")
            self.use_wandb = False

    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _compute_losses(self, out: dict) -> dict:
        """
        Compute all losses from a single forward pass output.

        FIX: u_R and u_t are now returned directly from
        generative.training_step() inside model.forward().
        No second sample_flow_path() call — that was the bug
        that caused the vector field prediction and its target
        to come from different random time samples.
        """
        return self.loss_fn(
            v_R_pred        = out["v_R_pred"],
            v_t_pred        = out["v_t_pred"],
            u_R             = out["u_R"],
            u_t             = out["u_t"],
            gen_coords      = out["gen_coords"],
            nmr_coords      = out["nmr_coords"],
            mu              = out["mu"],
            log_var         = out["log_var"],
            mask            = out["mask"],
            backbone_coords = out.get("target_coords"),
            atom_mask       = out.get("atom_mask"),
        )

    def _monitor_gradients(self) -> dict:
        """
        Compute per-module-group gradient L2 norms after backward().
        Called after unscale_ so norms reflect the true gradient magnitudes.
        Warns on vanishing (<1e-7) or exploding (>100) total norm.
        Returns dict suitable for wandb logging.
        """
        raw = self.model.module if self.multi_gpu else self.model

        group_sq = {"encoder": 0.0, "flow": 0.0, "generative": 0.0, "other": 0.0}
        total_sq  = 0.0
        n_none    = 0

        for name, param in raw.named_parameters():
            if param.grad is None:
                n_none += 1
                continue
            sq = param.grad.data.float().norm(2).item() ** 2
            total_sq += sq
            if "encoder" in name:
                group_sq["encoder"] += sq
            elif "flow" in name:
                group_sq["flow"] += sq
            elif "generative" in name or "decoder" in name:
                group_sq["generative"] += sq
            else:
                group_sq["other"] += sq

        stats = {f"grad_norm/{g}": group_sq[g] ** 0.5 for g in group_sq}
        total_norm = total_sq ** 0.5
        stats["grad_norm/total"] = total_norm

        if n_none > 0:
            stats["grad_norm/n_none"] = n_none

        if total_norm < 1e-7:
            logger.warning(
                f"Step {self.global_step}: vanishing gradients "
                f"(total_norm={total_norm:.2e})"
            )
        elif total_norm > 100.0:
            logger.warning(
                f"Step {self.global_step}: exploding gradients "
                f"(total_norm={total_norm:.2e})"
            )

        return stats

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        tcfg     = self.cfg["training"]
        log_every = self.cfg["logging"].get("log_every", 50)
        do_grad_monitor = (self.is_main and
                           self.global_step % log_every == 0)

        # Use correct dtype: bfloat16 for A100/H100/B200, float16 for consumer GPUs
        with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            out    = self.model(batch, n_gen=tcfg.get("n_gen_conformers", 10))
            losses = self._compute_losses(out)

        grad_stats = {}
        # bf16 doesn't need scaler.scale() — it never underflows
        if self.scaler.is_enabled():
            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.unscale_(self.optimizer)   # norms valid after unscale
            if do_grad_monitor:
                grad_stats = self._monitor_gradients()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), tcfg.get("max_grad_norm", 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses["total_loss"].backward()
            if do_grad_monitor:
                grad_stats = self._monitor_gradients()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), tcfg.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()

        self.scheduler.step()
        self._update_ema()

        result = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1
                  else v for k, v in losses.items()}
        result.update(grad_stats)
        return result

    def _update_ema(self):
        """Update EMA shadow weights after each optimizer step."""
        d = self.ema_decay
        raw = self.model.module if self.multi_gpu else self.model
        with torch.no_grad():
            for name, param in raw.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    self.ema_params[name].mul_(d).add_(
                        param.data.float(), alpha=1.0 - d
                    )

    def _apply_ema(self):
        """Swap live weights → EMA weights. Call before validation/inference."""
        raw = self.model.module if self.multi_gpu else self.model
        self._live_backup = {}
        for name, param in raw.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self._live_backup[name] = param.data.clone()
                param.data.copy_(self.ema_params[name].to(param.dtype))

    def _restore_live(self):
        """Restore live weights after EMA validation."""
        raw = self.model.module if self.multi_gpu else self.model
        for name, param in raw.named_parameters():
            if name in getattr(self, "_live_backup", {}):
                param.data.copy_(self._live_backup[name])
        self._live_backup = {}

    @torch.no_grad()
    def _val_step(self, batch: dict) -> dict:
        self.model.eval()
        tcfg = self.cfg["training"]
        out    = self.model(batch, n_gen=tcfg.get("n_gen_conformers", 10))
        losses = self._compute_losses(out)
        return {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1
                else v for k, v in losses.items()}

    def _log(self, losses: dict, phase: str = "train"):
        # Only rank-0 logs to avoid duplicate output in DDP
        if not self.is_main:
            return
        lcfg = self.cfg["logging"]
        keys = ["total_loss","flow_loss","ensemble_loss",
                "kl_loss","diversity_loss","geometry_loss"]
        msg  = " | ".join(f"{k.replace('_loss','')}:{losses.get(k,0):.4f}"
                           for k in keys if k in losses)
        # Append diversity diagnostics when available
        if "mean_pairwise_rmsd" in losses:
            msg += f" | prmsd:{losses['mean_pairwise_rmsd']:.2f}Å"
        logger.info(f"[{phase}] step {self.global_step:6d} | {msg}")

        if self.use_wandb:
            log_d = {f"{phase}/{k}": v for k, v in losses.items()
                     if isinstance(v, (int, float)) and not k.startswith("grad_norm/")}
            # Gradient norms logged at the top level (not prefixed by phase)
            for k, v in losses.items():
                if k.startswith("grad_norm/") and isinstance(v, (int, float)):
                    log_d[k] = v
            log_d["lr"] = self.scheduler.get_last_lr()[0]
            self.wandb.log(log_d, step=self.global_step)

    def _save_checkpoint(self, tag: str = "latest"):
        # Only rank-0 saves checkpoints
        if not self.is_main:
            return
        # Unwrap DDP to get raw state dict
        raw_model = self.model.module if self.multi_gpu else self.model
        ckpt = {
            "global_step":   self.global_step,
            "model_state":   raw_model.state_dict(),
            "ema_params":    self.ema_params,
            "optimizer":     self.optimizer.state_dict(),
            "scheduler_step":self.scheduler._step,
            "best_val_loss": self.best_val_loss,
            "config":        self.cfg,
        }
        path = self.checkpoint_dir / f"ckpt_{tag}.pt"
        torch.save(ckpt, str(path))
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler._step = ckpt.get("scheduler_step", 0)
        self.global_step     = ckpt.get("global_step", 0)
        self.best_val_loss   = ckpt.get("best_val_loss", float("inf"))
        if "ema_params" in ckpt:
            self.ema_params  = ckpt["ema_params"]
        logger.info(f"Resumed from {path} at step {self.global_step}")

    @torch.no_grad()
    def _validate(self, val_loader, max_batches: int = 50) -> dict:
        # Run validation on EMA weights — they are always smoother than live weights
        self._apply_ema()
        self.model.eval()
        all_losses = []
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            batch = self._to_device(batch)
            try:
                all_losses.append(self._val_step(batch))
            except Exception as e:
                logger.warning(f"Val step failed: {e}")

        if not all_losses:
            return {"total_loss": float("inf")}
        keys = all_losses[0].keys()
        result = {k: sum(d.get(k, 0) for d in all_losses
                         if isinstance(d.get(k), float)) / len(all_losses)
                  for k in keys}
        self._restore_live()
        return result

    def train(self, train_loader: DataLoader,
                    val_loader:   DataLoader):
        """Main training loop — works for single GPU and DDP multi-GPU."""
        tcfg = self.cfg["training"]

        # FIX 5f: For DDP, replace train_loader's sampler with DistributedSampler
        # so each rank gets a different subset of the data each epoch.
        if self.multi_gpu and self.world_size > 1:
            dist_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas = self.world_size,
                rank         = self.local_rank,
                shuffle      = True,
            )
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size  = train_loader.batch_size,
                sampler     = dist_sampler,
                num_workers = train_loader.num_workers,
                collate_fn  = train_loader.collate_fn,
                pin_memory  = True,
            )

        if self.is_main:
            logger.info(
                f"Training: epochs={tcfg['max_epochs']}  "
                f"batch={tcfg['batch_size']}  "
                f"struct={self.cfg['representation']['structure']}  "
                f"gen={self.cfg['generative_model']['type']}  "
                f"bf16={tcfg.get('use_bf16', False)}"
            )

        try:
            for epoch in range(tcfg["max_epochs"]):
                # Set epoch for DistributedSampler (ensures different shuffle each epoch)
                if self.multi_gpu and self.world_size > 1:
                    dist_sampler.set_epoch(epoch)

                if self.is_main:
                    logger.info(f"=== Epoch {epoch+1}/{tcfg['max_epochs']} ===")

                for batch in train_loader:
                    batch = self._to_device(batch)
                    try:
                        losses = self._train_step(batch)
                    except Exception as e:
                        logger.warning(
                            f"[rank {self.local_rank}] "
                            f"Step {self.global_step} failed: {e} — skipping"
                        )
                        self.global_step += 1
                        continue

                    self.global_step += 1

                    if (self.is_main and
                            self.global_step % self.cfg["logging"].get("log_every", 50) == 0):
                        self._log(losses, "train")

                    if (self.is_main and
                            self.global_step % tcfg.get("save_every", 2000) == 0):
                        self._save_checkpoint(f"step_{self.global_step}")

                # End-of-epoch validation — always runs regardless of val_every
                if self.is_main:
                    val_losses = self._validate(val_loader)
                    self._log(val_losses, "val")
                    if val_losses.get("total_loss", 999) < self.best_val_loss:
                        self.best_val_loss = val_losses["total_loss"]
                        self._save_checkpoint("best")
                        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
                    self._save_checkpoint("latest")

        finally:
            # Always cleanup DDP — even if training crashes
            cleanup_ddp()

        if self.is_main:
            logger.info("Training complete.")
            self._save_checkpoint("final")
