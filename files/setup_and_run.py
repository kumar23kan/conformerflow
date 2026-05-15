"""
ConformerFlow — Master Setup & Run Script
==========================================
Place this file in your conformerflow/ working directory and run:

    python setup_and_run.py

It will walk you through every step interactively:
  Step 0 — Check & install dependencies
  Step 1 — Set up directory structure
  Step 2 — Fetch NMR + X-ray PDB data from RCSB
  Step 3 — Parse NMR ensembles → tensors
  Step 4 — Parse held-out X-ray structures
  Step 5 — Filter & split dataset
  Step 6 — Train ConformerFlow
  Step 7 — Run inference on a structure
  Step 8 — Validate against NMR ground truth
"""

import os
import sys
import json
import time
import shutil
import subprocess
import platform
from pathlib import Path

# ─────────────────────────────────────────────────────────
# TERMINAL COLORS
# ─────────────────────────────────────────────────────────

class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║           ConformerFlow — Setup & Run Pipeline               ║
║   NMR-trained SE(3) Flow Matching for Protein Ensembles      ║
╚══════════════════════════════════════════════════════════════╝
{C.RESET}""")

def header(step: int, title: str):
    print(f"\n{C.BOLD}{C.BLUE}{'─'*65}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}  Step {step}: {title}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─'*65}{C.RESET}\n")

def info(msg: str):
    print(f"  {C.CYAN}ℹ  {msg}{C.RESET}")

def ok(msg: str):
    print(f"  {C.GREEN}✓  {msg}{C.RESET}")

def warn(msg: str):
    print(f"  {C.YELLOW}⚠  {msg}{C.RESET}")

def error(msg: str):
    print(f"  {C.RED}✗  {msg}{C.RESET}")

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"\n  {C.BOLD}{prompt}{suffix}: {C.RESET}").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    try:
        val = input(f"\n  {C.BOLD}{prompt} [{default_str}]: {C.RESET}").strip().lower()
        if not val:
            return default
        return val in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return default

def run_cmd(cmd: list, desc: str = "", check: bool = True,
            capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with nice output."""
    if desc:
        info(f"Running: {desc}")
    print(f"  {C.DIM}$ {' '.join(str(c) for c in cmd)}{C.RESET}")
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        error(f"Command failed (exit {result.returncode})")
        if result.stderr:
            print(f"  {C.RED}{result.stderr[:500]}{C.RESET}")
    return result


# ─────────────────────────────────────────────────────────
# STEP 0 — DEPENDENCIES
# ─────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("torch",        "torch"),
    ("biopython",    "Bio"),
    ("gemmi",        "gemmi"),
    ("requests",     "requests"),
    ("tqdm",         "tqdm"),
    ("numpy",        "numpy"),
    ("einops",       "einops"),
    ("scipy",        "scipy"),
]

OPTIONAL_PACKAGES = [
    ("fair-esm",     "esm",    "ESM-2 sequence embeddings (recommended)"),
    ("wandb",        "wandb",  "Training monitoring dashboard (optional)"),
]

def check_dependencies() -> dict:
    """Check which packages are installed."""
    status = {}
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            status[pip_name] = True
        except ImportError:
            status[pip_name] = False

    for pip_name, import_name, _ in OPTIONAL_PACKAGES:
        try:
            __import__(import_name)
            status[pip_name] = True
        except ImportError:
            status[pip_name] = False

    return status

def step0_dependencies():
    header(0, "Check & Install Dependencies")

    status = check_dependencies()

    # Show required packages
    print(f"  {C.BOLD}Required packages:{C.RESET}")
    missing_required = []
    for pip_name, _ in REQUIRED_PACKAGES:
        installed = status[pip_name]
        sym = f"{C.GREEN}✓{C.RESET}" if installed else f"{C.RED}✗{C.RESET}"
        print(f"    {sym}  {pip_name}")
        if not installed:
            missing_required.append(pip_name)

    print(f"\n  {C.BOLD}Optional packages:{C.RESET}")
    missing_optional = []
    for pip_name, _, desc in OPTIONAL_PACKAGES:
        installed = status[pip_name]
        sym = f"{C.GREEN}✓{C.RESET}" if installed else f"{C.YELLOW}○{C.RESET}"
        print(f"    {sym}  {pip_name:<12}  {C.DIM}{desc}{C.RESET}")
        if not installed:
            missing_optional.append(pip_name)

    # Install missing required
    if missing_required:
        warn(f"Missing required packages: {', '.join(missing_required)}")
        if ask_yes_no("Install missing required packages now?", default=True):
            run_cmd(
                [sys.executable, "-m", "pip", "install"] + missing_required,
                desc="Installing required packages"
            )
            ok("Required packages installed.")
        else:
            error("Cannot continue without required packages.")
            sys.exit(1)
    else:
        ok("All required packages installed.")

    # Install missing optional
    if missing_optional:
        if ask_yes_no(
            f"Install optional packages ({', '.join(missing_optional)})?",
            default=True
        ):
            for pkg in missing_optional:
                run_cmd(
                    [sys.executable, "-m", "pip", "install", pkg],
                    desc=f"Installing {pkg}"
                )
            ok("Optional packages installed.")
        else:
            warn("Optional packages skipped — some features may be unavailable.")

    # Detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            warn("No GPU detected — training will be slow on CPU.")
            warn("Consider using Google Colab or a cloud GPU instance.")
    except ImportError:
        pass

    return True


# ─────────────────────────────────────────────────────────
# STEP 1 — DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────

DIRS = [
    "data", "model", "training", "inference",
    "evaluation", "scripts", "configs",
    "pdb_data/nmr", "pdb_data/xray",
    "pdb_data/paired", "pdb_data/parsed_nmr",
    "pdb_data/parsed_xray", "pdb_data/splits",
    "checkpoints", "predictions", "validation_results",
]

INIT_FILES = [
    "data/__init__.py", "model/__init__.py",
    "training/__init__.py", "inference/__init__.py",
    "evaluation/__init__.py", "configs/__init__.py",
]

def step1_structure():
    header(1, "Set Up Directory Structure")

    base = Path(".")
    info(f"Working directory: {base.resolve()}")

    # Check source files exist
    expected_files = [
        "dataset.py", "encoder.py", "ensemble_stats.py",
        "fetch_pdb.py", "filter.py", "flow_matching.py",
        "frames.py", "losses.py", "metrics.py",
        "parse_nmr.py", "parse_xray.py", "train.py",
        "trainer.py", "validate.py",
    ]

    found = [f for f in expected_files if Path(f).exists()]
    missing = [f for f in expected_files if not Path(f).exists()]

    print(f"\n  {C.BOLD}Source files found: {len(found)}/{len(expected_files)}{C.RESET}")
    if missing:
        warn(f"Missing source files: {', '.join(missing)}")
        warn("Make sure all .py files are in the working directory.")

    # Create subdirectories
    print(f"\n  {C.BOLD}Creating directory structure:{C.RESET}")
    for d in DIRS:
        path = base / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"    {C.GREEN}+{C.RESET}  {d}/")

    # Create __init__.py files
    for init in INIT_FILES:
        p = base / init
        if not p.exists():
            p.touch()
            print(f"    {C.GREEN}+{C.RESET}  {init}")

    # File placement map: filename → target subfolder
    placement = {
        "dataset.py":        "data",
        "fetch_pdb.py":      "data",
        "filter.py":         "data",
        "parse_nmr.py":      "data",
        "parse_xray.py":     "data",
        "encoder.py":        "model",
        "encoders.py":       "model",
        "ensemble_stats.py": "model",
        "flow_matching.py":  "model",
        "frames.py":         "model",
        "generative_models.py": "model",
        "model_factory.py":  "model",
        "losses.py":         "training",
        "trainer.py":        "training",
        "train.py":          "scripts",
        "predict.py":        "inference",
        "sample.py":         "inference",
        "metrics.py":        "evaluation",
        "validate.py":       "evaluation",
    }

    print(f"\n  {C.BOLD}Moving files to subfolders:{C.RESET}")
    moved = 0
    for fname, folder in placement.items():
        src  = base / fname
        dest = base / folder / fname
        if src.exists() and not dest.exists():
            shutil.copy2(str(src), str(dest))
            print(f"    {C.CYAN}→{C.RESET}  {fname}  →  {folder}/")
            moved += 1
        elif dest.exists():
            print(f"    {C.DIM}=  {fname}  already in {folder}/{C.RESET}")

    ok(f"Directory structure ready. ({moved} files placed)")
    return True


# ─────────────────────────────────────────────────────────
# STEP 2 — FETCH PDB DATA
# ─────────────────────────────────────────────────────────

def step2_fetch_data(config: dict) -> bool:
    header(2, "Fetch NMR + X-ray PDB Data")

    # ── Custom NMR folder ──────────────────────────────────────────────────
    custom_dir = config.get("custom_nmr_dir", "").strip()
    if custom_dir:
        custom_path = Path(custom_dir)
        if not custom_path.exists():
            error(f"Custom NMR folder not found: {custom_dir}")
            return False

        pdb_files = list(custom_path.glob("*.pdb")) + list(custom_path.glob("*.PDB"))
        if not pdb_files:
            error(f"No .pdb files found in {custom_dir}")
            return False

        info(f"Custom NMR folder: {custom_dir}  ({len(pdb_files)} PDB files)")
        dest = Path("pdb_data/nmr")
        dest.mkdir(parents=True, exist_ok=True)

        if ask_yes_no(f"Copy {len(pdb_files)} files to pdb_data/nmr/?", default=True):
            for src in pdb_files:
                shutil.copy2(str(src), str(dest / src.name))
            ok(f"Copied {len(pdb_files)} NMR PDB files → pdb_data/nmr/")

        # Still fetch X-ray paired structures unless user skips
        info("Fetching held-out X-ray structures for validation (RCSB)...")
        max_xray = config.get("max_xray", 200)
        workers  = config.get("max_workers", 8)
        if ask_yes_no(f"Fetch up to {max_xray} X-ray paired structures from RCSB?",
                      default=True):
            cmd = [
                sys.executable, "data/fetch_pdb.py",
                "--output_dir",  "pdb_data",
                "--xray_only",
                "--max_xray",    str(max_xray),
                "--max_workers", str(workers),
            ]
            result = run_cmd(cmd, desc="Fetching X-ray entries from RCSB")
            if result.returncode != 0:
                warn("X-ray fetch failed — validation step may be unavailable.")
        return True

    # ── Standard RCSB fetch ────────────────────────────────────────────────
    info("This downloads NMR ensemble PDB files from RCSB.")
    info("~12,000 NMR entries = several hours depending on connection.")

    print(f"\n  {C.BOLD}Configuration:{C.RESET}")
    max_nmr  = config.get("max_nmr", 500)
    max_xray = config.get("max_xray", 200)
    workers  = config.get("max_workers", 8)
    print(f"    Max NMR entries:   {max_nmr}")
    print(f"    Max X-ray entries: {max_xray}")
    print(f"    Parallel workers:  {workers}")

    if not ask_yes_no("Start fetching PDB data?", default=True):
        warn("Skipping data fetch — using existing data if available.")
        return True

    cmd = [
        sys.executable, "data/fetch_pdb.py",
        "--output_dir",     "pdb_data",
        "--min_conformers", "5",
        "--max_nmr",        str(max_nmr),
        "--max_xray",       str(max_xray),
        "--max_workers",    str(workers),
    ]

    result = run_cmd(cmd, desc="Fetching PDB entries from RCSB")
    if result.returncode == 0:
        ok("PDB data fetched successfully.")
        for manifest in ["nmr_train.json", "paired.json"]:
            p = Path("pdb_data") / manifest
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                info(f"{manifest}: {len(data)} entries")
    else:
        error("Data fetch failed. Check your internet connection.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# STEP 3 — PARSE NMR ENSEMBLES
# ─────────────────────────────────────────────────────────

def step3_parse_nmr() -> bool:
    header(3, "Parse NMR Ensembles → Tensors")

    nmr_dir = Path("pdb_data/nmr")
    pdb_count = len(list(nmr_dir.glob("*.pdb"))) if nmr_dir.exists() else 0
    info(f"Found {pdb_count} NMR PDB files in pdb_data/nmr/")

    if pdb_count == 0:
        warn("No NMR PDB files found. Run Step 2 first.")
        if not ask_yes_no("Continue anyway?", default=False):
            return False

    if not ask_yes_no("Parse NMR ensembles now?", default=True):
        warn("Skipping NMR parsing.")
        return True

    cmd = [
        sys.executable, "data/parse_nmr.py",
        "--nmr_dir",        "pdb_data/nmr",
        "--output_dir",     "pdb_data/parsed_nmr",
        "--min_conformers", "5",
        "--max_residues",   "800",
    ]
    result = run_cmd(cmd, desc="Parsing NMR PDB files")
    if result.returncode == 0:
        parsed = list(Path("pdb_data/parsed_nmr").glob("*.npz"))
        ok(f"Parsed {len(parsed)} NMR ensembles → .npz tensor files.")
    else:
        error("NMR parsing failed.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# STEP 4 — PARSE X-RAY STRUCTURES
# ─────────────────────────────────────────────────────────

def step4_parse_xray() -> bool:
    header(4, "Parse Held-Out X-ray Structures")

    xray_dir  = Path("pdb_data/paired")
    pdb_count = len(list(xray_dir.glob("*.pdb"))) if xray_dir.exists() else 0
    info(f"Found {pdb_count} paired PDB files in pdb_data/paired/")

    if not ask_yes_no("Parse X-ray structures now?", default=True):
        warn("Skipping X-ray parsing.")
        return True

    cmd = [
        sys.executable, "data/parse_xray.py",
        "--xray_dir",    "pdb_data/paired",
        "--output_dir",  "pdb_data/parsed_xray",
        "--max_residues","800",
    ]
    result = run_cmd(cmd, desc="Parsing X-ray PDB files")
    if result.returncode == 0:
        parsed = list(Path("pdb_data/parsed_xray").glob("*.npz"))
        ok(f"Parsed {len(parsed)} X-ray structures → .npz tensor files.")
    else:
        error("X-ray parsing failed.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# STEP 5 — FILTER & SPLIT
# ─────────────────────────────────────────────────────────

def step5_filter_split() -> bool:
    header(5, "Quality Filter & Train/Val/Test Split")

    parsed_dir = Path("pdb_data/parsed_nmr")
    n_parsed   = len(list(parsed_dir.glob("*.npz"))) if parsed_dir.exists() else 0
    info(f"Found {n_parsed} parsed NMR tensors to filter.")

    paired_ids_json = Path("pdb_data/paired_nmr_ids.json")
    if paired_ids_json.exists():
        with open(paired_ids_json) as f:
            n_paired = len(json.load(f))
        info(f"Excluding {n_paired} paired NMR entries (held-out set).")
    else:
        warn("paired_nmr_ids.json not found — no entries will be excluded.")

    if not ask_yes_no("Run quality filtering and split?", default=True):
        warn("Skipping filter/split.")
        return True

    cmd = [
        sys.executable, "data/filter.py",
        "--parsed_dir",      "pdb_data/parsed_nmr",
        "--output_dir",      "pdb_data/splits",
        "--min_conformers",  "5",
        "--min_residues",    "20",
        "--max_residues",    "800",
        "--min_spread_rmsd", "0.1",
    ]
    if paired_ids_json.exists():
        cmd += ["--paired_ids_json", str(paired_ids_json)]

    result = run_cmd(cmd, desc="Filtering and splitting dataset")
    if result.returncode == 0:
        splits_dir = Path("pdb_data/splits")
        for split in ["train", "val", "test"]:
            p = splits_dir / f"{split}.json"
            if p.exists():
                with open(p) as f:
                    n = len(json.load(f))
                ok(f"{split:5s} split: {n:,} entries")
    else:
        error("Filter/split failed.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# GPU DETECTION — hardware-agnostic, handles any NVIDIA GPU
# A100, A800, H100, H800, B200 Blackwell, RTX consumer, etc.
# ─────────────────────────────────────────────────────────

# Known GPU profiles: (min_vram_gb, batch_size, d_model, secs_per_step)
# secs_per_step is an estimate for ConformerFlow at d_model=256, L=300
GPU_PROFILES = [
    # Blackwell B-series
    {"pattern": "B200",   "tier": "B200 Blackwell",  "vram": 192, "batch": 128, "d": 512, "sps": 0.05},
    {"pattern": "B100",   "tier": "B100 Blackwell",  "vram": 192, "batch": 128, "d": 512, "sps": 0.06},
    # Hopper H-series
    {"pattern": "H200",   "tier": "H200 SXM",        "vram": 141, "batch": 64,  "d": 512, "sps": 0.06},
    {"pattern": "H100",   "tier": "H100 SXM/PCIe",   "vram": 80,  "batch": 64,  "d": 512, "sps": 0.08},
    {"pattern": "H800",   "tier": "H800",             "vram": 80,  "batch": 64,  "d": 512, "sps": 0.08},
    # Ampere A-series datacenter
    {"pattern": "A800",   "tier": "A800 SXM",        "vram": 80,  "batch": 64,  "d": 512, "sps": 0.10},
    {"pattern": "A100",   "tier": "A100 SXM/PCIe",   "vram": 80,  "batch": 64,  "d": 512, "sps": 0.10},
    {"pattern": "A30",    "tier": "A30",              "vram": 24,  "batch": 16,  "d": 256, "sps": 0.20},
    {"pattern": "A10",    "tier": "A10/A10G",         "vram": 24,  "batch": 16,  "d": 256, "sps": 0.22},
    {"pattern": "A6000",  "tier": "RTX A6000",        "vram": 48,  "batch": 32,  "d": 384, "sps": 0.15},
    {"pattern": "A5000",  "tier": "RTX A5000",        "vram": 24,  "batch": 16,  "d": 256, "sps": 0.20},
    # Ada Lovelace L-series
    {"pattern": "L40",    "tier": "L40/L40S",        "vram": 48,  "batch": 32,  "d": 384, "sps": 0.12},
    {"pattern": "L4",     "tier": "L4",              "vram": 24,  "batch": 16,  "d": 256, "sps": 0.20},
    # Consumer Ada / Ampere RTX
    {"pattern": "4090",   "tier": "RTX 4090",        "vram": 24,  "batch": 16,  "d": 256, "sps": 0.18},
    {"pattern": "4080",   "tier": "RTX 4080",        "vram": 16,  "batch": 8,   "d": 256, "sps": 0.25},
    {"pattern": "4070",   "tier": "RTX 4070",        "vram": 12,  "batch": 6,   "d": 256, "sps": 0.30},
    {"pattern": "4060",   "tier": "RTX 4060",        "vram": 8,   "batch": 4,   "d": 256, "sps": 0.40},
    {"pattern": "3090",   "tier": "RTX 3090",        "vram": 24,  "batch": 16,  "d": 256, "sps": 0.22},
    {"pattern": "3080",   "tier": "RTX 3080",        "vram": 10,  "batch": 4,   "d": 256, "sps": 0.38},
    {"pattern": "3070",   "tier": "RTX 3070",        "vram": 8,   "batch": 4,   "d": 256, "sps": 0.42},
]

# VRAM-based fallback tiers (for unrecognised GPUs)
VRAM_TIERS = [
    (160, 128, 512, 0.05),   # >= 160 GB  (B200 class)
    (80,   64, 512, 0.08),   # >= 80 GB   (A100/H100 class)
    (48,   32, 384, 0.15),   # >= 48 GB   (A6000/L40 class)
    (24,   16, 256, 0.20),   # >= 24 GB   (A30/3090/4090 class)
    (16,    8, 256, 0.28),   # >= 16 GB   (4080 class)
    (8,     4, 256, 0.40),   # >= 8 GB    (4060/3070 class)
    (4,     2, 128, 1.00),   # >= 4 GB    (low-end)
    (0,     1, 128, 4.00),   # < 4 GB or CPU
]


def detect_gpu() -> dict:
    """
    Auto-detect GPU(s) and return hardware-appropriate training config.
    Works with any NVIDIA GPU including A800, B200 Blackwell, H100, etc.
    Also detects multi-GPU setups and adjusts batch size accordingly.
    """
    profile = {
        "has_gpu":       False,
        "gpu_count":     0,
        "gpu_names":     [],
        "gpu_tiers":     [],
        "total_vram_gb": 0.0,
        "vram_per_gpu":  0.0,
        "batch_size":    2,
        "d_model":       128,
        "secs_per_step": 5.0,
        "multi_gpu":     False,
        "notes":         [],
    }

    try:
        import torch
        if not torch.cuda.is_available():
            profile["notes"].append("No CUDA GPU detected — training on CPU.")
            return profile

        n_gpu = torch.cuda.device_count()
        profile["has_gpu"]   = True
        profile["gpu_count"] = n_gpu
        profile["multi_gpu"] = n_gpu > 1

        total_vram = 0.0
        for i in range(n_gpu):
            props      = torch.cuda.get_device_properties(i)
            name       = props.name
            vram_gb    = props.total_memory / 1e9
            total_vram += vram_gb
            profile["gpu_names"].append(name)

            # Match against known profiles
            matched = None
            name_upper = name.upper()
            for p in GPU_PROFILES:
                if p["pattern"].upper() in name_upper:
                    matched = p
                    break

            if matched:
                profile["gpu_tiers"].append(matched["tier"])
                if i == 0:  # use first GPU as primary reference
                    profile["batch_size"]    = matched["batch"]
                    profile["d_model"]       = matched["d"]
                    profile["secs_per_step"] = matched["sps"]
            else:
                # Fallback: use VRAM tiers
                profile["gpu_tiers"].append(f"Unknown ({vram_gb:.0f} GB VRAM)")
                if i == 0:
                    for min_v, bs, dm, sps in VRAM_TIERS:
                        if vram_gb >= min_v:
                            profile["batch_size"]    = bs
                            profile["d_model"]       = dm
                            profile["secs_per_step"] = sps
                            break

        profile["total_vram_gb"] = total_vram
        profile["vram_per_gpu"]  = total_vram / n_gpu

        # Multi-GPU: scale batch size linearly
        if n_gpu > 1:
            profile["batch_size"]    *= n_gpu
            profile["secs_per_step"] /= n_gpu
            profile["notes"].append(
                f"{n_gpu} GPUs detected — batch size scaled to {profile['batch_size']}."
            )
            profile["notes"].append(
                "Add --multi_gpu flag to training command for DDP."
            )

        # B200 / H100 specific notes
        for name in profile["gpu_names"]:
            nu = name.upper()
            if "B200" in nu or "B100" in nu:
                profile["notes"].append(
                    "Blackwell GPU detected — ensure CUDA 12.8+ and PyTorch 2.5+."
                )
            if "H100" in nu or "H800" in nu or "A100" in nu or "A800" in nu:
                profile["notes"].append(
                    "Datacenter GPU — enable bf16 training for best performance."
                )
                profile["notes"].append(
                    "Consider --compile flag (torch.compile) for extra speed."
                )

    except Exception as e:
        profile["notes"].append(f"GPU detection error: {e}")

    return profile


def print_gpu_profile(p: dict):
    """Print a clean GPU summary."""
    print(f"\n  {C.BOLD}Hardware detected:{C.RESET}")
    if not p["has_gpu"]:
        warn("No GPU — training on CPU only.")
        return

    for i, (name, tier) in enumerate(zip(p["gpu_names"], p["gpu_tiers"])):
        ok(f"GPU {i}: {name}  ({tier})")

    vram_str = (f"{p['total_vram_gb']:.0f} GB total"
                f"  ({p['vram_per_gpu']:.0f} GB per GPU)"
                if p["multi_gpu"]
                else f"{p['vram_per_gpu']:.0f} GB VRAM")
    info(f"VRAM:  {vram_str}")
    info(f"Suggested batch size: {p['batch_size']}")
    info(f"Suggested d_model:    {p['d_model']}")
    info(f"Est. seconds/step:    {p['secs_per_step']:.2f}s")

    for note in p["notes"]:
        warn(note)


# ─────────────────────────────────────────────────────────
# STEP 6 — TRAIN
# ─────────────────────────────────────────────────────────

def step6_train(config: dict) -> bool:
    header(6, "Train ConformerFlow")

    # Check data exists
    train_json = Path("pdb_data/splits/train.json")
    val_json   = Path("pdb_data/splits/val.json")

    if not train_json.exists() or not val_json.exists():
        error("Training data not found. Run Steps 2–5 first.")
        return False

    with open(train_json) as f: n_train = len(json.load(f))
    with open(val_json)   as f: n_val   = len(json.load(f))
    info(f"Training set:   {n_train:,} entries")
    info(f"Validation set: {n_val:,} entries")

    # Detect GPU and suggest settings
    gpu_profile = detect_gpu()
    has_gpu        = gpu_profile["has_gpu"]
    suggested_batch= gpu_profile["batch_size"]
    print_gpu_profile(gpu_profile)

    print(f"\n  {C.BOLD}Training configuration:{C.RESET}")
    batch_size  = config.get("batch_size",   suggested_batch)
    max_epochs  = config.get("max_epochs",   100)
    use_esm2    = config.get("use_esm2",     False)
    use_wandb   = config.get("use_wandb",    False)
    run_name    = config.get("run_name",     "conformerflow_run1")

    print(f"    Batch size:   {batch_size}")
    print(f"    Max epochs:   {max_epochs}")
    print(f"    ESM-2:        {'yes' if use_esm2 else 'no (one-hot only)'}")
    print(f"    WandB:        {'yes' if use_wandb else 'no'}")
    print(f"    Run name:     {run_name}")
    print()

    # Estimate training time using GPU-aware step timing
    steps_per_epoch = n_train // batch_size
    total_steps     = steps_per_epoch * max_epochs
    secs_per_step   = gpu_profile.get("secs_per_step", 0.5 if has_gpu else 5.0)
    est_hours       = (total_steps * secs_per_step) / 3600
    info(f"Estimated training time: ~{est_hours:.1f} hours  "
         f"({steps_per_epoch:,} steps/epoch x {max_epochs} epochs)")
    if est_hours > 48:
        warn("Very long job — use screen/tmux or a cluster scheduler.")
    elif est_hours > 8:
        warn("Long job — consider running overnight or in a tmux session.")

    if not ask_yes_no("Start training now?", default=True):
        warn("Skipping training.")
        info("To train later, run:")
        info("  python scripts/train.py --train_manifest pdb_data/splits/train.json \\")
        info("    --val_manifest pdb_data/splits/val.json")
        return True

    # FIX 3: all flags match train.py argparse names exactly
    # FIX 4: bf16 flag handled properly (no GradScaler for bf16)
    # FIX 5: multi-GPU uses torchrun + --multi_gpu flag for DDP
    cmd = [
        sys.executable, "scripts/train.py",
        "--config",           "configs/base_config.yaml",
        "--train_manifest",   "pdb_data/splits/train.json",
        "--val_manifest",     "pdb_data/splits/val.json",
        "--batch_size",       str(batch_size),
        "--max_epochs",       str(max_epochs),
        "--d_model",          str(config.get("d_model", 256)),
        "--output_dir",       "checkpoints",
        "--run_name",         run_name,
        # Architecture choices — names match train.py argparse exactly
        "--struct_repr",      config.get("struct_repr",      "se3_frames"),
        "--generative_model", config.get("generative_model", "flow_matching"),
        "--seq_encoder",      config.get("seq_encoder",      "onehot"),
        "--ensemble_stats",   config.get("ensemble_stats",   "full_covariance"),
        "--ode_method",       config.get("ode_method",       "heun"),
        "--loss_schedule",    config.get("loss_schedule",    "fixed"),
        # Data
        "--max_residues",     str(config.get("max_residues",  800)),
        "--max_conformers",   str(config.get("max_conformers", 50)),
        # Hyperparameters
        "--lr",               str(config.get("lr",            1e-4)),
        "--warmup_steps",     str(config.get("warmup_steps",  1000)),
        # Loss weights
        "--lambda_flow",      str(config.get("lambda_flow",      1.0)),
        "--lambda_ensemble",  str(config.get("lambda_ensemble",  1.0)),
        "--lambda_kl",        str(config.get("lambda_kl",        0.01)),
        "--lambda_diversity", str(config.get("lambda_diversity",  0.5)),
        "--lambda_geometry",  str(config.get("lambda_geometry",   0.1)),
    ]

    if use_esm2:  cmd.append("--use_esm2")
    if use_wandb: cmd.append("--use_wandb")

    if config.get("geom_warmup_epochs"):
        cmd += ["--geom_warmup_epochs", str(config["geom_warmup_epochs"])]
    if config.get("kl_anneal_epochs"):
        cmd += ["--kl_anneal_epochs",   str(config["kl_anneal_epochs"])]
    if config.get("curriculum_phase1"):
        cmd += ["--curriculum_phase1",  str(config["curriculum_phase1"]),
                "--curriculum_phase2",  str(config.get("curriculum_phase2", 40))]
    if config.get("pca_k"):
        cmd += ["--pca_k", str(config["pca_k"])]

    # FIX 4: bf16 for datacenter GPUs — passed as flag, Trainer handles autocast dtype
    datacenter_gpus = ["A100","A800","H100","H800","B200","B100","H200","L40"]
    if any(kw in " ".join(gpu_profile.get("gpu_names", [])).upper()
           for kw in datacenter_gpus):
        cmd.append("--bf16")
        info("Enabling bfloat16 mixed precision (datacenter GPU detected).")

    # FIX 5: multi-GPU — torchrun replaces python, --multi_gpu tells Trainer to init DDP
    if gpu_profile.get("multi_gpu"):
        n_gpu = gpu_profile["gpu_count"]
        cmd = (
            [sys.executable, "-m", "torch.distributed.run",
             f"--nproc_per_node={n_gpu}", "--master_port=29500"]
            + cmd[1:]
        )
        cmd.append("--multi_gpu")
        info(f"Launching DDP across {n_gpu} GPUs via torchrun.")

    result = run_cmd(cmd, desc="Training ConformerFlow")
    if result.returncode == 0:
        ok("Training complete! Checkpoint saved to checkpoints/")
    else:
        error("Training failed. Check error messages above.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# STEP 7 — INFERENCE
# ─────────────────────────────────────────────────────────

def step7_inference(config: dict) -> bool:
    header(7, "Run Inference — Generate Conformational Ensemble")

    # Find best checkpoint
    ckpt_dir  = Path("checkpoints")
    best_ckpt = ckpt_dir / "ckpt_best.pt"
    last_ckpt = ckpt_dir / "ckpt_latest.pt"

    if best_ckpt.exists():
        ckpt_path = best_ckpt
        ok(f"Using best checkpoint: {best_ckpt}")
    elif last_ckpt.exists():
        ckpt_path = last_ckpt
        warn(f"Best checkpoint not found — using latest: {last_ckpt}")
    else:
        ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        if ckpts:
            ckpt_path = ckpts[-1]
            warn(f"Using available checkpoint: {ckpt_path}")
        else:
            error("No checkpoint found. Run Step 6 (training) first.")
            return False

    # Find or ask for input structure
    default_input = config.get("inference_input", "")
    if not default_input:
        # Look for any PDB file nearby
        pdb_files = list(Path(".").glob("*.pdb")) + \
                    list(Path("pdb_data/paired").glob("*.pdb"))
        if pdb_files:
            default_input = str(pdb_files[0])

    input_pdb = ask(
        "Path to input PDB file for inference",
        default=default_input
    )
    if not input_pdb or not Path(input_pdb).exists():
        error(f"Input PDB not found: {input_pdb}")
        warn("Skipping inference step.")
        return True

    n_conformers = int(ask("Number of conformers to generate", default="20"))
    method       = ask("ODE method (euler/heun)", default="heun")
    n_steps      = int(ask("ODE integration steps (20=fast, 50=accurate)", default="20"))

    cmd = [
        sys.executable, "inference/sample.py",
        "--checkpoint",   str(ckpt_path),
        "--input",        input_pdb,
        "--n_conformers", str(n_conformers),
        "--n_steps",      str(n_steps),
        "--method",       method,
        "--output",       Path(input_pdb).stem + "_conformerflow",
    ]

    result = run_cmd(cmd, desc="Generating conformational ensemble")
    if result.returncode == 0:
        stem = Path(input_pdb).stem + "_conformerflow"
        ok(f"Ensemble generated!")
        ok(f"  {stem}_ensemble.pdb  — open in PyMOL / ChimeraX")
        ok(f"  {stem}_stats.json   — per-residue flexibility scores")
    else:
        error("Inference failed.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# STEP 8 — VALIDATION
# ─────────────────────────────────────────────────────────

def step8_validate(config: dict) -> bool:
    header(8, "Validate Against NMR Ground Truth")

    pairs_json = Path("pdb_data/paired.json")
    xray_dir   = Path("pdb_data/paired")
    nmr_dir    = Path("pdb_data/paired")

    if not pairs_json.exists():
        error("paired.json not found. Run Step 2 first.")
        return False

    with open(pairs_json) as f:
        pairs = json.load(f)
    info(f"Held-out paired proteins: {len(pairs)}")

    ckpt_dir  = Path("checkpoints")
    best_ckpt = ckpt_dir / "ckpt_best.pt"
    ckpts     = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    ckpt_path = best_ckpt if best_ckpt.exists() else (ckpts[-1] if ckpts else None)

    if not ckpt_path:
        error("No checkpoint found. Train the model first (Step 6).")
        return False

    ok(f"Using checkpoint: {ckpt_path}")

    max_proteins = int(ask(
        "How many proteins to validate? (0=all)",
        default=str(min(20, len(pairs)))
    ))

    if not ask_yes_no("Start validation?", default=True):
        warn("Skipping validation.")
        return True

    cmd = [
        sys.executable, "evaluation/validate.py",
        "--checkpoint",   str(ckpt_path),
        "--pairs_json",   str(pairs_json),
        "--xray_dir",     str(xray_dir),
        "--nmr_dir",      str(nmr_dir),
        "--output_dir",   "validation_results",
        "--n_conformers", str(config.get("n_conformers", 20)),
        "--n_steps",      "20",
        "--method",       "heun",
    ]
    if max_proteins > 0:
        cmd += ["--max_proteins", str(max_proteins)]

    result = run_cmd(cmd, desc="Running validation pipeline")
    if result.returncode == 0:
        summary_path = Path("validation_results/validation_summary.json")
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"\n  {C.BOLD}Key Results:{C.RESET}")
            key_metrics = [
                ("coverage_rmsd_mean",           "Coverage RMSD (Å)       ↓"),
                ("tm_score_mean",                 "TM-score                ↑"),
                ("rmsf_pearson_r_mean",           "RMSF Pearson r          ↑"),
                ("covariance_frobenius_sim_mean", "Covariance similarity   ↑"),
                ("clash_score_mean",              "Clash score             ↓"),
            ]
            for key, label in key_metrics:
                val = summary.get(key)
                if val is not None:
                    print(f"    {label}: {val:.4f}")
        ok("Validation complete! Results in validation_results/")
    else:
        error("Validation failed.")
        return False
    return True


# ─────────────────────────────────────────────────────────
# MEMORY BUDGET CHECKER
# Warns before training if batch × residues × conformers
# will exceed available VRAM — catches silent OOM crashes.
# ─────────────────────────────────────────────────────────

def check_memory_budget(batch_size:    int,
                         max_residues:  int,
                         max_conformers:int,
                         d_model:       int,
                         vram_gb:       float) -> dict:
    """
    Estimate peak training memory usage and warn if unsafe.

    Dominant term is the attention matrix inside the encoder:
      batch × conformers × heads × L² × 4 bytes (fp32)
    Plus activations, gradients, optimizer states (~6× model params).

    Returns dict with estimated_gb, safe (bool), and warning message.
    """
    bytes_per_elem  = 4   # fp32; bf16 halves this
    n_heads         = 8

    # Attention matrix  (B * M * H * L * L * 4 bytes)
    attn_mem = (batch_size * max_conformers * n_heads
                * max_residues * max_residues * bytes_per_elem)

    # Coordinate tensors  (B * M * L * 4_atoms * 3 * 4 bytes)
    coord_mem = batch_size * max_conformers * max_residues * 4 * 3 * bytes_per_elem

    # Model parameters + gradients + Adam states (~6x param memory)
    # Rough param count at given d_model
    approx_params  = 10_000_000 * (d_model / 256) ** 2
    param_mem      = approx_params * bytes_per_elem * 6

    # Activations + gradient checkpoints add ~3x overhead during backprop
    TRAINING_OVERHEAD = 3.5
    total_bytes = (attn_mem + coord_mem + param_mem) * TRAINING_OVERHEAD
    est_gb      = total_bytes / 1e9

    # Safety threshold: 85% of available VRAM
    safe_limit = vram_gb * 0.85 if vram_gb > 0 else 8.0
    is_safe    = est_gb <= safe_limit

    msg = (f"Estimated peak VRAM: {est_gb:.1f} GB  "
           f"(limit: {safe_limit:.1f} GB = 85% of {vram_gb:.0f} GB)")

    return {
        "estimated_gb": est_gb,
        "safe":         is_safe,
        "safe_limit":   safe_limit,
        "message":      msg,
    }


def suggest_safe_residues(batch_size:    int,
                           max_conformers:int,
                           d_model:       int,
                           vram_gb:       float) -> int:
    """
    Binary-search for the largest max_residues that fits in VRAM.
    Returns a safe value rounded down to nearest 50.
    """
    lo, hi = 50, 2000
    while lo < hi - 50:
        mid = (lo + hi) // 2
        budget = check_memory_budget(
            batch_size, mid, max_conformers, d_model, vram_gb
        )
        if budget["safe"]:
            lo = mid
        else:
            hi = mid
    return (lo // 50) * 50   # round to nearest 50


# ─────────────────────────────────────────────────────────
# THREE CONFIGURATION MODES
# ─────────────────────────────────────────────────────────

def configure_auto(gpu_profile: dict) -> dict:
    """
    AUTO mode — zero questions asked.
    Detects GPU, sets everything optimally, runs end-to-end.
    Best for: first run, cluster jobs, scripted pipelines.
    """
    print(f"\n  {C.BOLD}{C.GREEN}AUTO mode — detecting optimal settings...{C.RESET}\n")

    vram = gpu_profile.get("vram_per_gpu", 0)

    # Safe max_residues for auto mode
    safe_res = suggest_safe_residues(
        batch_size     = gpu_profile["batch_size"],
        max_conformers = 50,
        d_model        = gpu_profile["d_model"],
        vram_gb        = vram,
    ) if vram > 0 else 400

    config = {
        # Data
        "max_nmr":       15000,
        "max_xray":      5000,
        "max_workers":   8,
        "max_residues":  safe_res,
        "max_conformers":50,
        # Model
        "batch_size":    gpu_profile["batch_size"],
        "d_model":       gpu_profile["d_model"],
        "max_epochs":    100,
        # Features
        "use_esm2":      _esm2_available(),
        "use_wandb":     False,
        "run_name":      "conformerflow_auto",
        "n_conformers":  20,
        # Architecture — all defaults
        "struct_repr":      "se3_frames",
        "generative_model": "flow_matching",
        "seq_encoder":      "esm2_650m" if _esm2_available() else "onehot",
        "ensemble_stats":   "full_covariance",
        "ode_method":       "heun",
        "loss_schedule":    "fixed",
        # Mode
        "auto_confirm":  True,
    }

    _print_config_summary(config, gpu_profile, mode="AUTO")
    return config


def configure_guided(gpu_profile: dict) -> dict:
    """
    GUIDED mode — shows GPU-suggested defaults, user can accept or override.
    Each parameter has a one-line explanation of what it does.
    Best for: first-time users who want to understand what they're setting.
    """
    print(f"\n  {C.BOLD}GUIDED mode — press Enter to accept each suggestion.{C.RESET}")
    print(f"  {C.DIM}GPU-aware defaults shown in brackets.{C.RESET}\n")

    vram = gpu_profile.get("vram_per_gpu", 0)
    config = {}

    # ── Data ──
    print(f"  {C.BOLD}── Data settings ──{C.RESET}")

    # Custom NMR folder
    custom_dir = ask(
        "Custom NMR folder path  [leave blank to download from RCSB]",
        default=""
    ).strip()
    config["custom_nmr_dir"] = custom_dir
    if custom_dir:
        ok(f"Will use local NMR files from: {custom_dir}")
        config["max_nmr"]  = 0   # not used when custom folder is set
        config["max_xray"] = int(ask("Max X-ray entries to fetch for validation",
                                     default="200"))
        config["max_workers"] = int(ask("Parallel download workers", default="8"))
    else:
        scale = ask(
            "Data scale  [full=12k NMR entries | small=500 for testing]",
            default="full"
        )
        if scale == "small":
            config["max_nmr"], config["max_xray"] = 500, 200
            warn("Small scale is for pipeline testing only.")
        elif scale == "full":
            config["max_nmr"], config["max_xray"] = 15000, 5000
        else:
            config["max_nmr"]  = int(ask("Max NMR entries", default="5000"))
            config["max_xray"] = int(ask("Max X-ray entries", default="2000"))

        config["max_workers"] = int(ask(
            "Parallel download workers  [more = faster fetch, respect rate limits]",
            default="8"
        ))

    # ── Model + memory ──
    print(f"\n  {C.BOLD}── Model settings ──{C.RESET}")

    config["batch_size"] = int(ask(
        f"Batch size  [GPU suggested: {gpu_profile['batch_size']}]",
        default=str(gpu_profile["batch_size"])
    ))
    config["d_model"] = int(ask(
        f"Model width d_model  [128=fast/small | 256=full | 512=large datacenter GPU]"
        f"  [GPU suggested: {gpu_profile['d_model']}]",
        default=str(gpu_profile["d_model"])
    ))
    config["max_conformers"] = int(ask(
        "Max conformers per NMR entry  [higher=richer training signal, more memory]",
        default="50"
    ))

    # Max residues — compute safe suggestion and show memory budget
    safe_res = suggest_safe_residues(
        config["batch_size"], config["max_conformers"],
        config["d_model"], vram
    ) if vram > 0 else 600

    budget = check_memory_budget(
        config["batch_size"], safe_res,
        config["max_conformers"], config["d_model"], vram
    )
    print(f"\n  {C.DIM}Memory estimate at L={safe_res}: "
          f"{budget['estimated_gb']:.1f} GB / {vram:.0f} GB VRAM{C.RESET}")

    config["max_residues"] = int(ask(
        f"Max residues per protein  [attention scales as L², safe suggestion: {safe_res}]",
        default=str(safe_res)
    ))

    # Re-check budget with user's choice
    final_budget = check_memory_budget(
        config["batch_size"], config["max_residues"],
        config["max_conformers"], config["d_model"], vram
    )
    if not final_budget["safe"] and vram > 0:
        warn(final_budget["message"])
        warn("This combination may cause an out-of-memory crash.")
        if not ask_yes_no("Proceed anyway?", default=False):
            config["max_residues"] = safe_res
            ok(f"Reverted to safe value: max_residues={safe_res}")
    else:
        ok(final_budget["message"])

    # ── Training ──
    print(f"\n  {C.BOLD}── Training settings ──{C.RESET}")

    config["max_epochs"] = int(ask(
        "Max epochs  [100 is standard; more = better but longer]",
        default="100"
    ))

    config["use_esm2"] = False
    if _esm2_available():
        config["use_esm2"] = ask_yes_no(
            "Use ESM-2 embeddings?  [better accuracy, +2 GB VRAM, slower startup]",
            default=True
        )
    else:
        warn("ESM-2 not installed — using one-hot only. Install: pip install fair-esm")

    config["use_wandb"] = False
    if _wandb_available():
        config["use_wandb"] = ask_yes_no(
            "Log to Weights & Biases?  [live training charts in browser]",
            default=False
        )

    config["run_name"]      = ask("Run name  [identifies this experiment]",
                                   default="conformerflow_run1")
    config["n_conformers"]  = int(ask(
        "Conformers to generate at inference  [user-specified N]",
        default="20"
    ))

    # ── Architecture (optional, just the two most impactful) ──
    print(f"\n  {C.BOLD}── Architecture  (optional){C.RESET}")
    print(f"  {C.DIM}Defaults are SE(3) frames + flow matching — the current best settings.{C.RESET}")
    if ask_yes_no("Change structural representation or generative model?", default=False):
        config["struct_repr"]      = _select_arch_choice("struct_repr",      show_all=False)
        config["generative_model"] = _select_arch_choice("generative_model", show_all=False)
    else:
        config["struct_repr"]      = "se3_frames"
        config["generative_model"] = "flow_matching"

    # Apply remaining arch defaults
    for key in ("ensemble_stats", "ode_method", "loss_schedule", "seq_encoder"):
        config[key] = ARCH_CHOICES[key]["default"]

    config["auto_confirm"]  = False

    _print_config_summary(config, gpu_profile, mode="GUIDED")
    return config


# ─────────────────────────────────────────────────────────
# ARCHITECTURE CHOICE CATALOGUE
# All choices default to the current best-known setting.
# In MANUAL mode every axis is exposed. In GUIDED mode only
# the most impactful ones are shown. AUTO never asks.
# ─────────────────────────────────────────────────────────

ARCH_CHOICES = {

    # ── Structural representation ──────────────────────────
    "struct_repr": {
        "label":   "Structural representation",
        "default": "se3_frames",
        "options": {
            "se3_frames": (
                "SE(3) frames  [current default]",
                "Rotation matrix + translation per residue. Equivariant by "
                "construction — the model output is valid regardless of how "
                "the protein is oriented in space. Most principled choice. "
                "Requires Gram-Schmidt at each decoder step."
            ),
            "cartesian": (
                "Cartesian coordinates",
                "Raw xyz of Cα atoms. Simplest to implement. NOT rotation-"
                "invariant — the model must learn symmetry from data, which "
                "costs capacity. Good baseline for ablation."
            ),
            "distance_matrix": (
                "Distance matrix  (pairwise Cα distances)",
                "Fully rotation- and translation-invariant. Loses chirality "
                "(mirror images look identical). Cannot directly generate 3D "
                "coordinates — needs a separate reconstruction step (e.g. MDS). "
                "Good for understanding covariance structure."
            ),
            "torsion_angles": (
                "Backbone torsion angles  (φ, ψ, ω)",
                "Compact (3 values per residue vs 3×3+3 for SE(3)). Naturally "
                "rotation-invariant. Directly encodes chain geometry. Requires "
                "a backbone reconstruction pass (Nerf/FrameDiff style) to get "
                "3D coordinates at inference. Fast to train."
            ),
        },
        "impact": "Changes what the model can learn and how equivariance is enforced.",
    },

    # ── Generative backbone ────────────────────────────────
    "generative_model": {
        "label":   "Generative model",
        "default": "flow_matching",
        "options": {
            "flow_matching": (
                "Conditional Flow Matching  (CFM)  [current default]",
                "Straight-line ODE paths from noise to data. Fast inference "
                "(20 steps), modern, stable training. Best choice for new work."
            ),
            "ot_cfm": (
                "Optimal Transport CFM  (OT-CFM)",
                "Uses mini-batch OT to find shorter flow paths. Faster "
                "convergence than standard CFM, same inference cost. "
                "Recommended if standard CFM shows slow training loss descent."
            ),
            "ddpm": (
                "DDPM diffusion",
                "Original denoising diffusion. Well-studied, large literature "
                "to compare against. Slower inference (1000 steps). Best "
                "choice if you want to compare against published diffusion "
                "protein models (RFDiffusion, FrameDiff)."
            ),
            "ddim": (
                "DDIM diffusion  (deterministic)",
                "Deterministic version of DDPM. Same training as DDPM but "
                "50–100× faster inference. Good if you train with DDPM but "
                "want fast inference."
            ),
            "vae": (
                "Variational Autoencoder  (VAE)",
                "Encoder maps ensemble to latent μ/σ, decoder reconstructs. "
                "Most interpretable latent space — you can interpolate between "
                "conformations in latent space. Lower sample quality than flow "
                "matching. Good for studying latent structure."
            ),
            "score_matching": (
                "Score matching  (denoising score matching)",
                "Trains the model to predict the score (gradient of log "
                "density). Closely related to diffusion but different training "
                "signal. Some evidence it captures multimodal distributions "
                "better than flow matching for proteins."
            ),
        },
        "impact": "Major scientific choice — different inductive biases about conformational space.",
    },

    # ── Sequence encoder ───────────────────────────────────
    "seq_encoder": {
        "label":   "Sequence encoder",
        "default": "esm2_650m",
        "options": {
            "onehot": (
                "One-hot only  (20-dimensional)",
                "No pretrained knowledge. Fast. Good for testing whether "
                "geometry alone is sufficient to predict flexibility. "
                "Useful ablation: does sequence information matter?"
            ),
            "esm2_650m": (
                "ESM-2 650M  [current default]",
                "Strong general protein language model (650M parameters). "
                "Captures evolutionary information, secondary structure, "
                "conservation. Needs ~2 GB extra VRAM. Recommended."
            ),
            "esm2_3b": (
                "ESM-2 3B",
                "More powerful ESM-2 variant (3B parameters). Better "
                "representation of rare sequences and disorder. Needs ~8 GB "
                "extra VRAM. Use only on A100/H100/B200 class GPUs."
            ),
            "prot_t5": (
                "ProtT5-XL",
                "Alternative protein language model trained on UniRef50. "
                "Different training data and architecture from ESM-2. "
                "Good for comparing whether ESM-2 features are specifically "
                "useful or any PLM would work."
            ),
            "no_sequence": (
                "No sequence encoding  (geometry only)",
                "Feed only structural coordinates, no sequence. Pure ablation: "
                "does the model learn purely from backbone geometry? Tests "
                "whether sequence is necessary for flexibility prediction."
            ),
        },
        "impact": "Controls how much evolutionary and biochemical context the model has.",
    },

    # ── Ensemble statistics ────────────────────────────────
    "ensemble_stats": {
        "label":   "Ensemble statistics representation",
        "default": "full_covariance",
        "options": {
            "full_covariance": (
                "Mean + full covariance matrix  [current default]",
                "Captures correlated motions between all residue pairs "
                "(L×L matrix). Most expressive. Expensive: O(L²) memory "
                "and compute. Captures hinge bending, domain movements."
            ),
            "diagonal": (
                "Mean + diagonal variance",
                "Treats each residue independently. O(L) memory. Much faster. "
                "Misses correlated motions entirely. Good for initial training "
                "or if you believe flexibility is mostly local."
            ),
            "top_k_pca": (
                "Mean + top-k PCA modes",
                "Computes top-k eigenvectors of the covariance matrix. "
                "Captures dominant collective motions efficiently. Good "
                "tradeoff between expressiveness and cost. k=10–50 covers "
                "most biologically relevant motions."
            ),
            "conformer_attention": (
                "Raw conformer cross-attention  (let the transformer learn)",
                "No explicit statistics — the flow transformer attends "
                "directly to all M conformers at each step. Most flexible "
                "but most expensive. The model learns what statistics matter."
            ),
        },
        "impact": "Controls whether correlated residue motions are captured in the distribution.",
    },

    # ── ODE integrator ─────────────────────────────────────
    "ode_method": {
        "label":   "ODE integration method  (inference)",
        "default": "heun",
        "options": {
            "euler": (
                "Euler  (1st order)",
                "Simplest. 20 steps. Fastest inference. Lower quality — "
                "can drift from the true trajectory on complex proteins."
            ),
            "heun": (
                "Heun  (2nd order RK)  [current default]",
                "Best balance of speed and quality. 20 steps. Each step "
                "evaluates the vector field twice (predictor-corrector). "
                "Recommended for most use cases."
            ),
            "rk4": (
                "RK4  (4th order)",
                "Most accurate. 50 steps, 4 evaluations per step. Use when "
                "Heun produces physically implausible conformers."
            ),
            "adaptive": (
                "Adaptive step size  (dopri5 via torchdiffeq)",
                "Automatically controls step size to meet error tolerance. "
                "Variable number of function evaluations per trajectory. "
                "Best for rigorously correct conformers regardless of cost."
            ),
        },
        "impact": "Accuracy vs speed tradeoff at inference time only — does not affect training.",
    },

    # ── Loss schedule ──────────────────────────────────────
    "loss_schedule": {
        "label":   "Loss weight schedule",
        "default": "fixed",
        "options": {
            "fixed": (
                "Fixed weights throughout training  [current default]",
                "All five loss weights stay constant. Simple, predictable. "
                "Good starting point for any new experiment."
            ),
            "geometry_warmup": (
                "Geometry warmup  (anneal geometry loss down)",
                "Start with high geometry weight (force valid bonds early), "
                "then anneal down as training progresses and the model learns "
                "natural geometry from data. Reduces geometry loss interference "
                "with learning conformational diversity later in training."
            ),
            "kl_annealing": (
                "KL annealing  (ramp KL weight up slowly)",
                "Start with KL weight = 0, ramp up over first N epochs. "
                "Prevents posterior collapse early in training. Standard "
                "technique for VAE-style models with flow matching."
            ),
            "curriculum": (
                "Curriculum  (geometry→ensemble→diversity)",
                "Phase 1: geometry + flow only (learn valid structures). "
                "Phase 2: add ensemble loss (match NMR statistics). "
                "Phase 3: add diversity loss (span conformational space). "
                "Most principled but requires careful epoch budgeting."
            ),
        },
        "impact": "Controls what the model prioritises at each stage of training.",
    },
}


def _select_arch_choice(key: str, show_all: bool = True) -> str:
    """
    Interactive selector for a single architecture axis.
    Shows all options with their scientific tradeoff descriptions.
    Returns the chosen option key.
    """
    meta    = ARCH_CHOICES[key]
    default = meta["default"]
    options = meta["options"]

    print(f"\n  {C.BOLD}{meta['label']}{C.RESET}")
    print(f"  {C.DIM}Impact: {meta['impact']}{C.RESET}\n")

    keys = list(options.keys())
    for i, (opt_key, (short, long_desc)) in enumerate(options.items()):
        marker = f"{C.GREEN}*{C.RESET}" if opt_key == default else " "
        print(f"  {marker} {i+1})  {C.CYAN}{opt_key}{C.RESET}  —  {short}")
        if show_all:
            # Wrap long description at 70 chars
            words = long_desc.split()
            line, lines = [], []
            for w in words:
                line.append(w)
                if len(" ".join(line)) > 68:
                    lines.append(" ".join(line[:-1]))
                    line = [w]
            if line:
                lines.append(" ".join(line))
            for l in lines:
                print(f"       {C.DIM}{l}{C.RESET}")
        print()

    choice = ask(
        f"Choose {meta['label']}  [1–{len(keys)}, or type key name]",
        default=default
    ).strip()

    # Accept number or key name
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    if choice in options:
        return choice

    warn(f"Unrecognised choice '{choice}' — using default: {default}")
    return default


def configure_manual(gpu_profile: dict) -> dict:
    """
    MANUAL mode — full control over every parameter.
    Shows the allowed range and impact for each setting.
    Best for: researchers who know exactly what they want.
    """
    print(f"\n  {C.BOLD}MANUAL mode — full control over all parameters.{C.RESET}")
    print(f"  {C.DIM}Allowed ranges and impacts shown for each setting.{C.RESET}\n")

    vram   = gpu_profile.get("vram_per_gpu", 0)
    config = {}

    # ── Data ──
    print(f"  {C.BOLD}── Data parameters ──{C.RESET}")
    custom_dir = ask(
        "Custom NMR folder path  [absolute path to your local NMR PDB files, "
        "or leave blank to download from RCSB]",
        default=""
    ).strip()
    config["custom_nmr_dir"] = custom_dir
    if custom_dir:
        ok(f"Will use local NMR files from: {custom_dir}")
        config["max_nmr"] = 0
    else:
        config["max_nmr"] = int(ask(
            "Max NMR entries to fetch  [range: 100–20000 | more = richer training set]",
            default="15000"
        ))
    config["max_xray"] = int(ask(
        "Max X-ray entries to fetch  [range: 100–10000 | used for held-out validation]",
        default="5000"
    ))
    config["max_workers"] = int(ask(
        "Parallel download workers  [range: 1–16 | RCSB rate limit ~10/s recommended]",
        default="8"
    ))

    # ── Memory-critical parameters ──
    print(f"\n  {C.BOLD}── Memory-critical parameters ──{C.RESET}")
    print(f"  {C.DIM}Peak VRAM = batch × conformers × L² × heads × 4 bytes{C.RESET}")
    print(f"  {C.DIM}Available VRAM: {vram:.0f} GB{C.RESET}\n")

    config["batch_size"] = int(ask(
        "Batch size  [range: 1–256 | larger = more stable gradients, more VRAM]",
        default=str(gpu_profile["batch_size"])
    ))
    config["max_conformers"] = int(ask(
        "Max conformers per NMR entry  [range: 5–100 | all M used simultaneously]",
        default="50"
    ))
    config["max_residues"] = int(ask(
        "Max residues per protein  [range: 50–2000 | attention is O(L²) — big impact]",
        default="800"
    ))

    # Memory check with full detail
    budget = check_memory_budget(
        config["batch_size"], config["max_residues"],
        config["max_conformers"],
        gpu_profile["d_model"], vram
    )
    sym = f"{C.GREEN}✓{C.RESET}" if budget["safe"] else f"{C.RED}✗{C.RESET}"
    print(f"\n  {sym}  {budget['message']}")
    if not budget["safe"] and vram > 0:
        warn("Unsafe combination — likely OOM during training.")
        safe_res = suggest_safe_residues(
            config["batch_size"], config["max_conformers"],
            gpu_profile["d_model"], vram
        )
        info(f"Safe max_residues at these settings: {safe_res}")
        if ask_yes_no(f"Override max_residues to {safe_res}?", default=True):
            config["max_residues"] = safe_res

    # ── Model architecture ──
    print(f"\n  {C.BOLD}── Model architecture ──{C.RESET}")
    config["d_model"] = int(ask(
        "d_model  [128=fast/debug | 256=standard | 384=large | 512=datacenter only]",
        default=str(gpu_profile["d_model"])
    ))
    config["n_encoder_layers"] = int(ask(
        "Encoder layers  [range: 2–8 | more = stronger representation]",
        default="4"
    ))
    config["n_flow_layers"] = int(ask(
        "Flow transformer layers  [range: 3–12 | more = better generative quality]",
        default="8"
    ))

    # ── Training ──
    print(f"\n  {C.BOLD}── Training hyperparameters ──{C.RESET}")
    config["max_epochs"]   = int(ask(
        "Max epochs  [range: 10–500]",
        default="100"
    ))
    config["lr"]           = float(ask(
        "Learning rate  [range: 1e-5 – 1e-3 | 1e-4 is standard for transformers]",
        default="1e-4"
    ))
    config["warmup_steps"] = int(ask(
        "Warmup steps  [range: 100–5000 | linear LR ramp at start of training]",
        default="1000"
    ))

    # ── Loss weights ──
    print(f"\n  {C.BOLD}── Loss weights ──{C.RESET}")
    print(f"  {C.DIM}Total = flow + ensemble + kl + diversity + geometry{C.RESET}")
    config["lambda_flow"]      = float(ask(
        "lambda_flow      [flow matching: core SE(3) learning signal]",
        default="1.0"
    ))
    config["lambda_ensemble"]  = float(ask(
        "lambda_ensemble  [match NMR distribution statistics]",
        default="1.0"
    ))
    config["lambda_kl"]        = float(ask(
        "lambda_kl        [latent space regularisation — keep small]",
        default="0.01"
    ))
    config["lambda_diversity"] = float(ask(
        "lambda_diversity [prevent mode collapse — all conformers identical]",
        default="0.5"
    ))
    config["lambda_geometry"]  = float(ask(
        "lambda_geometry  [CA bond lengths + clash penalty]",
        default="0.1"
    ))

    # ── Features ──
    print(f"\n  {C.BOLD}── Optional features ──{C.RESET}")
    config["use_esm2"]  = ask_yes_no(
        "Use ESM-2 embeddings?  [+accuracy, +2 GB VRAM]",
        default=_esm2_available()
    ) if _esm2_available() else False

    config["use_wandb"] = ask_yes_no(
        "Log to Weights & Biases?",
        default=False
    ) if _wandb_available() else False

    config["run_name"]     = ask("Run name", default="conformerflow_manual")
    config["n_conformers"] = int(ask(
        "Conformers at inference  [user-specified N — any value works]",
        default="20"
    ))

    # ── Architecture choices ──
    print(f"\n  {C.BOLD}── Architecture choices ──{C.RESET}")
    print(f"  {C.DIM}These are the major scientific design decisions.{C.RESET}")
    print(f"  {C.DIM}Defaults are the current best-known settings.{C.RESET}")
    print(f"  {C.DIM}Change them to run ablations or test alternative approaches.{C.RESET}\n")

    if ask_yes_no("Configure architecture choices? (N = use all defaults)", default=False):
        config["struct_repr"]      = _select_arch_choice("struct_repr")
        config["generative_model"] = _select_arch_choice("generative_model")
        config["seq_encoder"]      = _select_arch_choice("seq_encoder")
        config["ensemble_stats"]   = _select_arch_choice("ensemble_stats")
        config["ode_method"]       = _select_arch_choice("ode_method")
        config["loss_schedule"]    = _select_arch_choice("loss_schedule")

        # ── Loss weight annealing schedule params ──
        sched = config.get("loss_schedule", "fixed")
        if sched == "geometry_warmup":
            config["geom_warmup_epochs"] = int(ask(
                "Anneal geometry loss over how many epochs?",
                default="20"
            ))
        elif sched == "kl_annealing":
            config["kl_anneal_epochs"] = int(ask(
                "Ramp KL weight up over how many epochs?",
                default="30"
            ))
        elif sched == "curriculum":
            config["curriculum_phase1"] = int(ask(
                "Phase 1 epochs  (geometry + flow only)",
                default="20"
            ))
            config["curriculum_phase2"] = int(ask(
                "Phase 2 epochs  (+ ensemble loss)",
                default="40"
            ))
            print(f"  {C.DIM}Phase 3 (+ diversity) runs for remaining epochs.{C.RESET}")

        # ── top-k PCA if chosen ──
        if config.get("ensemble_stats") == "top_k_pca":
            config["pca_k"] = int(ask(
                "Number of PCA modes k  [range: 5–100 | 10–50 covers most motions]",
                default="20"
            ))

        # ── ESM-2 model size if chosen ──
        if config.get("seq_encoder") in ("esm2_650m", "esm2_3b"):
            config["use_esm2"] = True
        elif config.get("seq_encoder") == "prot_t5":
            config["use_esm2"] = False
            config["use_prot_t5"] = True
        else:
            config["use_esm2"] = False

    else:
        # Apply all defaults
        for key, meta in ARCH_CHOICES.items():
            config[key] = meta["default"]
        config["use_esm2"] = _esm2_available()
        info("Using default architecture (SE(3) frames + CFM + ESM-2 + full covariance).")

    config["auto_confirm"] = False

    _print_config_summary(config, gpu_profile, mode="MANUAL")
    return config


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _esm2_available() -> bool:
    try:
        import esm
        return True
    except ImportError:
        return False


def _wandb_available() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


def _print_config_summary(config: dict, gpu_profile: dict, mode: str):
    """Print a clean summary of the chosen configuration."""
    print(f"\n  {C.BOLD}{C.BLUE}{'─'*60}{C.RESET}")
    print(f"  {C.BOLD}{C.BLUE}  Configuration summary  [{mode}]{C.RESET}")
    print(f"  {C.BOLD}{C.BLUE}{'─'*60}{C.RESET}")

    groups = [
        ("Data", [
            ("NMR entries",        config.get("max_nmr",        "auto")),
            ("X-ray entries",      config.get("max_xray",       "auto")),
            ("Max residues",       config.get("max_residues",   "auto")),
            ("Max conformers",     config.get("max_conformers", "auto")),
        ]),
        ("Model", [
            ("d_model",            config.get("d_model",        gpu_profile.get("d_model", 256))),
            ("Batch size",         config.get("batch_size",     gpu_profile.get("batch_size", 4))),
            ("Epochs",             config.get("max_epochs",     100)),
            ("ESM-2",              "yes" if config.get("use_esm2") else "no"),
            ("WandB",              "yes" if config.get("use_wandb") else "no"),
        ]),
        ("Architecture", [
            ("Struct repr",        config.get("struct_repr",      "se3_frames")),
            ("Generative model",   config.get("generative_model", "flow_matching")),
            ("Sequence encoder",   config.get("seq_encoder",      "esm2_650m")),
            ("Ensemble stats",     config.get("ensemble_stats",   "full_covariance")),
            ("ODE method",         config.get("ode_method",       "heun")),
            ("Loss schedule",      config.get("loss_schedule",    "fixed")),
        ]),
        ("Run", [
            ("Name",               config.get("run_name",      "conformerflow")),
            ("Conformers (infer)", config.get("n_conformers",  20)),
            ("Auto-confirm steps", "yes" if config.get("auto_confirm") else "no"),
        ]),
    ]
    for group_name, items in groups:
        print(f"\n  {C.BOLD}{group_name}:{C.RESET}")
        for label, value in items:
            # Highlight non-default architecture choices
            is_arch   = group_name == "Architecture"
            arch_key  = {"Struct repr":"struct_repr","Generative model":"generative_model",
                         "Sequence encoder":"seq_encoder","Ensemble stats":"ensemble_stats",
                         "ODE method":"ode_method","Loss schedule":"loss_schedule"}.get(label)
            is_default= (arch_key and
                         ARCH_CHOICES.get(arch_key, {}).get("default") == value)
            color = C.DIM + C.CYAN if (is_arch and is_default) else C.CYAN
            tag   = "" if not is_arch else (f" {C.DIM}(default){C.RESET}" if is_default
                                            else f" {C.YELLOW}← changed{C.RESET}")
            print(f"    {label:<26} {color}{value}{C.RESET}{tag}")
    print()


# ─────────────────────────────────────────────────────────
# STEP SELECTOR
# ─────────────────────────────────────────────────────────

ALL_STEPS = [
    (0, "Check & install dependencies"),
    (1, "Set up directory structure"),
    (2, "Fetch NMR + X-ray PDB data"),
    (3, "Parse NMR ensembles"),
    (4, "Parse X-ray structures"),
    (5, "Quality filter & split dataset"),
    (6, "Train ConformerFlow"),
    (7, "Run inference on a structure"),
    (8, "Validate against NMR ground truth"),
]


def select_steps(auto_run: bool = False) -> list:
    """
    Let user choose which steps to run.
    In auto mode, always runs all steps 0-8.
    """
    if auto_run:
        info("Auto mode: running all steps 0-8.")
        return list(range(9))

    print(f"\n{C.BOLD}  Available steps:{C.RESET}")
    for num, desc in ALL_STEPS:
        print(f"    {C.CYAN}{num}{C.RESET}  {desc}")

    print()
    print(f"  {C.DIM}Examples: '0-8' = all  |  '0,1,2' = setup only  |  "
          f"'6' = train only  |  '6,7,8' = train + infer + validate{C.RESET}")
    choice = ask("Which steps to run?", default="0-8")

    selected = set()
    for part in choice.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            selected.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            selected.add(int(part))

    return sorted(selected)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    banner()

    print(f"  {C.DIM}Python {sys.version.split()[0]}  |  "
          f"Platform: {platform.system()} {platform.machine()}{C.RESET}")
    print(f"  {C.DIM}Working directory: {Path('.').resolve()}{C.RESET}\n")

    # ── Mode selection ──
    print(f"  {C.BOLD}Choose a run mode:{C.RESET}\n")
    print(f"    {C.GREEN}A{C.RESET}  AUTO      — detect GPU, set everything optimally, run end-to-end")
    print(f"             {C.DIM}(no questions, best for cluster / first run){C.RESET}")
    print(f"    {C.CYAN}G{C.RESET}  GUIDED    — see GPU suggestions, accept or override each setting")
    print(f"             {C.DIM}(recommended for most users){C.RESET}")
    print(f"    {C.YELLOW}M{C.RESET}  MANUAL    — full control: every parameter, memory check, loss weights")
    print(f"             {C.DIM}(for researchers who know exactly what they want){C.RESET}\n")

    mode = ask("Mode", default="G").strip().upper()
    if mode not in ("A", "AUTO", "G", "GUIDED", "M", "MANUAL"):
        mode = "G"
    mode = mode[0]   # A / G / M

    # ── GPU detection (always runs) ──
    gpu_profile = detect_gpu()

    # ── Build config from chosen mode ──
    if mode == "A":
        config         = configure_auto(gpu_profile)
        selected_steps = select_steps(auto_run=True)
    elif mode == "M":
        selected_steps = select_steps(auto_run=False)
        config         = configure_manual(gpu_profile)
    else:
        selected_steps = select_steps(auto_run=False)
        config         = configure_guided(gpu_profile)

    if not selected_steps:
        warn("No steps selected. Exiting.")
        return

    auto_confirm = config.get("auto_confirm", False)
    if auto_confirm:
        print(f"\n{C.BOLD}{C.GREEN}  AUTO mode — running all steps without confirmation.{C.RESET}\n")
    else:
        print(f"\n{C.BOLD}{C.GREEN}  Starting ConformerFlow pipeline...{C.RESET}")
        print(f"  {C.DIM}Steps selected: {selected_steps}{C.RESET}\n")

    # ── Patch step functions to respect auto_confirm ──
    # In auto mode, all ask_yes_no prompts default to True (no input needed).
    # We achieve this by monkey-patching the module-level ask_yes_no.
    import setup_and_run as _self
    if auto_confirm:
        _self.ask_yes_no = lambda prompt, default=True: True
        _self.ask        = lambda prompt, default="": default

    step_fns = {
        0: lambda: step0_dependencies(),
        1: lambda: step1_structure(),
        2: lambda: step2_fetch_data(config),
        3: lambda: step3_parse_nmr(),
        4: lambda: step4_parse_xray(),
        5: lambda: step5_filter_split(),
        6: lambda: step6_train(config),
        7: lambda: step7_inference(config),
        8: lambda: step8_validate(config),
    }

    results    = {}
    start_time = time.time()

    for step_num in selected_steps:
        fn = step_fns.get(step_num)
        if fn is None:
            warn(f"Unknown step {step_num} — skipping.")
            continue

        try:
            success = fn()
            results[step_num] = "✓" if success else "✗"
        except KeyboardInterrupt:
            warn(f"\nStep {step_num} interrupted.")
            results[step_num] = "⚠"
            if not auto_confirm:
                if ask_yes_no("Continue with remaining steps?", default=True):
                    continue
            break
        except Exception as e:
            error(f"Step {step_num} failed: {e}")
            results[step_num] = "✗"
            if not auto_confirm:
                if ask_yes_no("Continue with remaining steps?", default=True):
                    continue
            break

    # ── Final summary ──
    elapsed = time.time() - start_time
    print(f"\n{C.BOLD}{C.BLUE}{'═'*65}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}  ConformerFlow Pipeline Summary{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'═'*65}{C.RESET}")

    step_names = {
        0:"Dependencies",  1:"Directory setup", 2:"Fetch data",
        3:"Parse NMR",     4:"Parse X-ray",     5:"Filter & split",
        6:"Training",      7:"Inference",        8:"Validation",
    }
    for step_num, status in results.items():
        color = C.GREEN if status == "✓" else C.RED if status == "✗" else C.YELLOW
        name  = step_names.get(step_num, f"Step {step_num}")
        print(f"  {color}{status}{C.RESET}  Step {step_num}: {name}")

    print(f"\n  {C.DIM}Total time: {elapsed/60:.1f} minutes{C.RESET}")

    print(f"\n  {C.BOLD}Output locations:{C.RESET}")
    outputs = [
        ("pdb_data/splits/",         "Training data splits"),
        ("checkpoints/ckpt_best.pt", "Best trained model"),
        ("predictions/",             "Generated ensembles"),
        ("validation_results/",      "Evaluation metrics"),
    ]
    for path, desc in outputs:
        exists = Path(path).exists()
        sym = f"{C.GREEN}✓{C.RESET}" if exists else f"{C.DIM}○{C.RESET}"
        print(f"    {sym}  {path:<36} {C.DIM}{desc}{C.RESET}")

    print(f"\n{C.BOLD}{C.GREEN}  ConformerFlow pipeline complete.{C.RESET}\n")


if __name__ == "__main__":
    main()
