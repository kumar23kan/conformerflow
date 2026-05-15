"""
ConformerFlow — Training Entry Point
Reads YAML config, validates it, applies CLI overrides, builds model, trains.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/base_config.yaml
    python scripts/train.py --struct_repr torsions --generative_model ddpm
    python scripts/train.py --batch_size 64 --bf16 --multi_gpu
"""

import sys, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_validator import load_and_validate, set_nested
from data.dataset             import build_dataloaders
from training.trainer         import Trainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train ConformerFlow — all flags override the YAML config"
    )

    # ── Config file ──
    parser.add_argument("--config",         default="configs/base_config.yaml")
    parser.add_argument("--no_auto_batch",  action="store_true")

    # ── Architecture choices ──
    parser.add_argument("--struct_repr",
        choices=["se3_frames","cartesian","distances","torsions"])
    parser.add_argument("--generative_model",
        choices=["flow_matching","ddpm","ddim","vae","score_matching"])
    parser.add_argument("--seq_encoder",
        choices=["onehot","esm2_650m","esm2_3b","prot_t5","none"])
    parser.add_argument("--ensemble_stats",
        choices=["full_covariance","diagonal","pca_topk","attention"])
    parser.add_argument("--ode_method",
        choices=["heun","euler","rk4"])
    parser.add_argument("--loss_schedule",
        choices=["fixed","geometry_warmup","kl_annealing","curriculum"])

    # ── Data ──
    parser.add_argument("--max_residues",    type=int)
    parser.add_argument("--max_conformers",  type=int)
    parser.add_argument("--train_manifest",  type=str)
    parser.add_argument("--val_manifest",    type=str)

    # ── Model ──
    parser.add_argument("--d_model",          type=int)
    parser.add_argument("--n_encoder_layers", type=int)
    parser.add_argument("--n_flow_layers",    type=int)
    parser.add_argument("--pca_k",            type=int)

    # ── Training ──
    parser.add_argument("--batch_size",      type=int)
    parser.add_argument("--max_epochs",      type=int)
    parser.add_argument("--lr",              type=float)
    parser.add_argument("--warmup_steps",    type=int)
    parser.add_argument("--output_dir",      type=str)
    parser.add_argument("--run_name",        type=str)
    parser.add_argument("--resume_from",     type=str)

    # ── Loss weights ──
    parser.add_argument("--lambda_flow",      type=float)
    parser.add_argument("--lambda_ensemble",  type=float)
    parser.add_argument("--lambda_kl",        type=float)
    parser.add_argument("--lambda_diversity", type=float)
    parser.add_argument("--lambda_geometry",  type=float)

    # ── Loss schedule params ──
    parser.add_argument("--geom_warmup_epochs",  type=int)
    parser.add_argument("--kl_anneal_epochs",    type=int)
    parser.add_argument("--curriculum_phase1",   type=int)
    parser.add_argument("--curriculum_phase2",   type=int)

    # ── Features & hardware ──
    parser.add_argument("--use_esm2",  action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--bf16",      action="store_true",
                        help="bfloat16 mixed precision (A100/H100/B200)")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="DistributedDataParallel across all GPUs")

    args = parser.parse_args()

    # ── Load and validate base config ──
    cfg = load_and_validate(
        args.config,
        auto_batch = not args.no_auto_batch,
        verbose    = True,
    )

    # ── Apply all CLI overrides → nested config ──
    overrides = [
        (args.struct_repr,        "representation.structure"),
        (args.generative_model,   "generative_model.type"),
        (args.seq_encoder,        "representation.sequence"),
        (args.ensemble_stats,     "ensemble_stats.covariance"),
        (args.ode_method,         "generative_model.ode_method"),
        (args.loss_schedule,      "training.loss_schedule"),
        (args.max_residues,       "data.max_residues"),
        (args.max_conformers,     "training.n_gen_conformers"),
        (args.d_model,            "model.d_model"),
        (args.n_encoder_layers,   "model.n_encoder_layers"),
        (args.n_flow_layers,      "model.n_flow_layers"),
        (args.pca_k,              "ensemble_stats.n_pca_modes"),
        (args.batch_size,         "training.batch_size"),
        (args.max_epochs,         "training.max_epochs"),
        (args.lr,                 "training.learning_rate"),
        (args.warmup_steps,       "training.warmup_steps"),
        (args.output_dir,         "training.checkpoint_dir"),
        (args.run_name,           "logging.run_name"),
        (args.resume_from,        "training.resume_from"),
        (args.lambda_flow,        "loss.lambda_flow"),
        (args.lambda_ensemble,    "loss.lambda_ensemble"),
        (args.lambda_kl,          "loss.lambda_kl"),
        (args.lambda_diversity,   "loss.lambda_diversity"),
        (args.lambda_geometry,    "loss.lambda_geometry"),
        (args.geom_warmup_epochs, "training.geom_warmup_epochs"),
        (args.kl_anneal_epochs,   "training.kl_anneal_epochs"),
        (args.curriculum_phase1,  "training.curriculum_phase1"),
        (args.curriculum_phase2,  "training.curriculum_phase2"),
    ]
    for value, path in overrides:
        if value is not None:
            set_nested(cfg, path, value)
            logger.info(f"CLI override: {path} = {value}")

    # ESM-2 shorthand
    if args.use_esm2 and cfg["representation"]["sequence"] == "onehot":
        set_nested(cfg, "representation.sequence", "esm2_650m")

    if args.use_wandb:
        set_nested(cfg, "logging.use_wandb", True)

    # Hardware flags stored in training section
    cfg["training"]["use_bf16"]  = args.bf16
    cfg["training"]["multi_gpu"] = args.multi_gpu

    # ── Manifest paths ──
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    train_manifest = (args.train_manifest or
                      f"{data_cfg['pdb_data_dir']}/splits/train.json")
    val_manifest   = (args.val_manifest or
                      f"{data_cfg['pdb_data_dir']}/splits/val.json")

    # ── Data loaders ──
    logger.info("Building data loaders...")
    loaders = build_dataloaders(
        train_manifest = train_manifest,
        val_manifest   = val_manifest,
        test_manifest  = val_manifest,
        batch_size     = train_cfg["batch_size"],
        max_residues   = data_cfg["max_residues"],
        max_conformers = train_cfg.get("n_gen_conformers", 50),
        num_workers    = data_cfg.get("max_workers", 4),
    )

    # ── Train ──
    trainer = Trainer(cfg)
    trainer.train(loaders["train"], loaders["val"])


if __name__ == "__main__":
    main()
