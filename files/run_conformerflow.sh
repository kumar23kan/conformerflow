#!/bin/bash
# ============================================================
#  ConformerFlow — Full Pipeline Runner
#  Place this script in your conformerflow/ working folder
#  and run:  bash run_conformerflow.sh
# ============================================================

set -e  # stop on any error

# ── Colors for terminal output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

# ── Logging helpers ──
log_info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
log_step()    { echo -e "\n${BOLD}${CYAN}━━━  $1  ━━━${NC}"; }
log_banner()  {
    echo -e "${BOLD}${CYAN}"
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║          ConformerFlow Pipeline Runner           ║"
    echo "  ║   NMR-trained SE(3) Conformational Ensemble AI  ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ============================================================
#  USER CONFIGURATION — edit these before running
# ============================================================

CONDA_ENV="conformerflow"      # conda environment name
PDB_DATA_DIR="./pdb_data"      # where PDB files are downloaded
MAX_NMR=15000                  # max NMR entries to fetch
MAX_XRAY=5000                  # max X-ray entries to fetch
MIN_CONFORMERS=5               # minimum NMR conformers required
MAX_RESIDUES=800               # maximum sequence length
BATCH_SIZE=4                   # training batch size (reduce if OOM)
MAX_EPOCHS=100                 # training epochs
N_CONFORMERS=20                # conformers to generate at inference
CHECKPOINT_DIR="./checkpoints" # where model checkpoints are saved
USE_ESM2=false                 # set to true if fair-esm is installed
USE_WANDB=false                # set to true if wandb is installed
MAX_WORKERS=8                  # parallel download threads

# ── Colors for step status tracking ──
STEP_LOG="./pipeline_progress.log"

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

check_python() {
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.11+ or activate your conda env."
    fi
    PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $PY_VER"
}

check_gpu() {
    GPU_INFO=$(python -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f'CUDA  {n}x {name}  ({mem} GB VRAM)')
else:
    print('CPU only')
" 2>/dev/null || echo "torch not installed yet")
    log_info "GPU: $GPU_INFO"
}

check_file_structure() {
    log_info "Checking directory structure..."
    REQUIRED_DIRS=("data" "model" "training" "inference" "evaluation" "scripts")
    MISSING=0
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            log_warn "Missing directory: $dir/ — creating it"
            mkdir -p "$dir"
            touch "$dir/__init__.py"
            MISSING=1
        fi
    done

    REQUIRED_FILES=(
        "data/dataset.py" "data/fetch_pdb.py" "data/filter.py"
        "data/parse_nmr.py" "data/parse_xray.py"
        "model/encoder.py" "model/ensemble_stats.py"
        "model/flow_matching.py" "model/frames.py"
        "training/losses.py" "training/trainer.py"
        "scripts/train.py"
    )
    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$f" ]; then
            log_error "Missing required file: $f  — make sure all ConformerFlow files are in place"
        fi
    done

    # Ensure __init__.py files exist
    for dir in data model training inference evaluation scripts; do
        [ -d "$dir" ] && touch "$dir/__init__.py"
    done

    log_success "Directory structure OK"
}

step_done() {
    local step="$1"
    echo "$step" >> "$STEP_LOG"
}

step_already_done() {
    local step="$1"
    [ -f "$STEP_LOG" ] && grep -qx "$step" "$STEP_LOG"
}

confirm_step() {
    local msg="$1"
    echo -e "${YELLOW}  → $msg${NC}"
    read -r -p "    Proceed? [Y/n/skip] " choice
    choice="${choice:-Y}"
    case "$choice" in
        [Yy]*) return 0 ;;
        [Ss]*) return 1 ;;   # skip
        *)     log_warn "Skipping."; return 1 ;;
    esac
}

show_eta() {
    local msg="$1"
    echo -e "  ${YELLOW}⏱  Estimated time: $msg${NC}"
}

# ============================================================
#  STEP 0 — ENVIRONMENT SETUP
# ============================================================

run_step0_environment() {
    log_step "STEP 0: Environment Setup"

    if step_already_done "step0"; then
        log_info "Step 0 already completed — skipping"
        return
    fi

    check_python

    log_info "Installing required Python packages..."
    show_eta "2–5 minutes"

    pip install --upgrade pip -q

    # Core packages
    pip install torch --index-url https://download.pytorch.org/whl/cu118 -q \
        2>/dev/null || pip install torch -q

    pip install \
        biopython \
        gemmi \
        requests \
        tqdm \
        numpy \
        pandas \
        einops \
        scipy \
        -q

    # Optional: ESM-2
    if [ "$USE_ESM2" = true ]; then
        log_info "Installing ESM-2 (fair-esm)..."
        pip install fair-esm -q && log_success "ESM-2 installed" \
            || log_warn "ESM-2 install failed — will use one-hot encoding only"
    fi

    # Optional: WandB
    if [ "$USE_WANDB" = true ]; then
        log_info "Installing wandb..."
        pip install wandb -q && log_success "WandB installed" \
            || log_warn "WandB install failed — will log to console only"
    fi

    # Verify torch
    python -c "import torch; print(f'  PyTorch {torch.__version__} installed')" \
        || log_error "PyTorch installation failed"

    check_gpu
    log_success "Environment setup complete"
    step_done "step0"
}

# ============================================================
#  STEP 1 — FETCH PDB DATA
# ============================================================

run_step1_fetch() {
    log_step "STEP 1: Fetch PDB Data from RCSB"

    if step_already_done "step1"; then
        log_info "Step 1 already completed — skipping"
        return
    fi

    if ! confirm_step "Download up to $MAX_NMR NMR + $MAX_XRAY X-ray PDB files from RCSB (~8-15 GB)"; then
        log_warn "Step 1 skipped"
        return
    fi

    show_eta "2–6 hours (RCSB rate-limited) — safe to run overnight"

    mkdir -p "$PDB_DATA_DIR"

    python data/fetch_pdb.py \
        --output_dir     "$PDB_DATA_DIR" \
        --min_conformers "$MIN_CONFORMERS" \
        --max_nmr        "$MAX_NMR" \
        --max_xray       "$MAX_XRAY" \
        --max_workers    "$MAX_WORKERS"

    # Verify outputs
    NMR_COUNT=$(ls "$PDB_DATA_DIR/nmr/"*.pdb 2>/dev/null | wc -l)
    log_success "Downloaded $NMR_COUNT NMR PDB files"
    step_done "step1"
}

# ============================================================
#  STEP 2 — PARSE NMR STRUCTURES
# ============================================================

run_step2_parse_nmr() {
    log_step "STEP 2: Parse NMR Ensemble Structures"

    if step_already_done "step2"; then
        log_info "Step 2 already completed — skipping"
        return
    fi

    if ! confirm_step "Parse all NMR PDB files → .npz tensor files (~20-40 GB)"; then
        log_warn "Step 2 skipped"
        return
    fi

    show_eta "30–90 minutes"

    mkdir -p "$PDB_DATA_DIR/parsed_nmr"

    python data/parse_nmr.py \
        --nmr_dir        "$PDB_DATA_DIR/nmr" \
        --output_dir     "$PDB_DATA_DIR/parsed_nmr" \
        --min_conformers "$MIN_CONFORMERS" \
        --max_residues   "$MAX_RESIDUES"

    PARSED_COUNT=$(ls "$PDB_DATA_DIR/parsed_nmr/"*.npz 2>/dev/null | wc -l)
    log_success "Parsed $PARSED_COUNT NMR ensembles"
    step_done "step2"
}

# ============================================================
#  STEP 3 — PARSE X-RAY STRUCTURES (paired held-out set)
# ============================================================

run_step3_parse_xray() {
    log_step "STEP 3: Parse Paired X-ray Structures (Held-out Validation Set)"

    if step_already_done "step3"; then
        log_info "Step 3 already completed — skipping"
        return
    fi

    if ! confirm_step "Parse paired X-ray PDB files for held-out validation"; then
        log_warn "Step 3 skipped"
        return
    fi

    show_eta "5–15 minutes"

    mkdir -p "$PDB_DATA_DIR/parsed_xray"

    python data/parse_xray.py \
        --xray_dir    "$PDB_DATA_DIR/paired" \
        --output_dir  "$PDB_DATA_DIR/parsed_xray" \
        --max_residues "$MAX_RESIDUES"

    XRAY_COUNT=$(ls "$PDB_DATA_DIR/parsed_xray/"*.npz 2>/dev/null | wc -l)
    log_success "Parsed $XRAY_COUNT X-ray structures"
    step_done "step3"
}

# ============================================================
#  STEP 4 — FILTER & SPLIT DATASET
# ============================================================

run_step4_filter() {
    log_step "STEP 4: Quality Filter & Train/Val/Test Split"

    if step_already_done "step4"; then
        log_info "Step 4 already completed — skipping"
        return
    fi

    if ! confirm_step "Apply quality filters and split dataset into train/val/test"; then
        log_warn "Step 4 skipped"
        return
    fi

    show_eta "5–10 minutes"

    mkdir -p "$PDB_DATA_DIR/splits"

    PAIRED_IDS_JSON="$PDB_DATA_DIR/paired_nmr_ids.json"
    PAIRED_FLAG=""
    if [ -f "$PAIRED_IDS_JSON" ]; then
        PAIRED_FLAG="--paired_ids_json $PAIRED_IDS_JSON"
    fi

    python data/filter.py \
        --parsed_dir     "$PDB_DATA_DIR/parsed_nmr" \
        --output_dir     "$PDB_DATA_DIR/splits" \
        $PAIRED_FLAG \
        --min_conformers "$MIN_CONFORMERS" \
        --min_residues   20 \
        --max_residues   "$MAX_RESIDUES" \
        --min_spread_rmsd 0.1

    # Report split sizes
    for split in train val test; do
        if [ -f "$PDB_DATA_DIR/splits/$split.json" ]; then
            COUNT=$(python -c "import json; d=json.load(open('$PDB_DATA_DIR/splits/$split.json')); print(len(d))")
            log_info "  $split set: $COUNT entries"
        fi
    done

    log_success "Dataset ready"
    step_done "step4"
}

# ============================================================
#  STEP 5 — TRAIN THE MODEL
# ============================================================

run_step5_train() {
    log_step "STEP 5: Train ConformerFlow"

    if step_already_done "step5"; then
        log_info "Step 5 already completed — skipping"
        log_info "To retrain, remove 'step5' from $STEP_LOG"
        return
    fi

    if ! confirm_step "Train ConformerFlow on NMR ensembles (epochs=$MAX_EPOCHS, batch=$BATCH_SIZE)"; then
        log_warn "Step 5 skipped"
        return
    fi

    # Auto-adjust batch size based on GPU memory
    GPU_MEM=$(python -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(gb)
else:
    print(0)
" 2>/dev/null || echo "0")

    if [ "$GPU_MEM" -ge 24 ] 2>/dev/null; then
        BATCH_SIZE=8
        log_info "24+ GB GPU detected — using batch_size=8"
    elif [ "$GPU_MEM" -ge 16 ] 2>/dev/null; then
        BATCH_SIZE=4
        log_info "16 GB GPU detected — using batch_size=4"
    elif [ "$GPU_MEM" -ge 8 ] 2>/dev/null; then
        BATCH_SIZE=2
        log_info "8 GB GPU detected — using batch_size=2"
    elif [ "$GPU_MEM" -eq 0 ] 2>/dev/null; then
        BATCH_SIZE=1
        log_warn "No GPU detected — training on CPU (very slow)"
    fi

    show_eta "12–48 hours depending on GPU"

    WANDB_FLAG=""
    [ "$USE_WANDB" = true ] && WANDB_FLAG="--use_wandb"

    ESM2_FLAG=""
    [ "$USE_ESM2" = true ] && ESM2_FLAG="--use_esm2"

    python scripts/train.py \
        --train_manifest "$PDB_DATA_DIR/splits/train.json" \
        --val_manifest   "$PDB_DATA_DIR/splits/val.json" \
        --batch_size     "$BATCH_SIZE" \
        --max_epochs     "$MAX_EPOCHS" \
        --output_dir     "$CHECKPOINT_DIR" \
        --run_name       "conformerflow_$(date +%Y%m%d_%H%M)" \
        --warmup_steps   1000 \
        --val_every      500 \
        --save_every     2000 \
        --n_gen_conformers 10 \
        $ESM2_FLAG \
        $WANDB_FLAG

    log_success "Training complete"
    step_done "step5"
}

# ============================================================
#  STEP 6 — INFERENCE (run on a sample structure)
# ============================================================

run_step6_inference() {
    log_step "STEP 6: Generate Conformational Ensemble (Inference)"

    if step_already_done "step6"; then
        log_info "Step 6 already completed"
        log_info "To predict a new structure, run:"
        echo "  python inference/sample.py \\"
        echo "    --checkpoint $CHECKPOINT_DIR/ckpt_best.pt \\"
        echo "    --input your_structure.pdb \\"
        echo "    --n_conformers $N_CONFORMERS"
        return
    fi

    # Find best checkpoint
    CKPT=""
    for candidate in \
        "$CHECKPOINT_DIR/ckpt_best.pt" \
        "$CHECKPOINT_DIR/ckpt_final.pt" \
        "$CHECKPOINT_DIR/ckpt_latest.pt"; do
        if [ -f "$candidate" ]; then
            CKPT="$candidate"
            break
        fi
    done

    if [ -z "$CKPT" ]; then
        log_warn "No checkpoint found in $CHECKPOINT_DIR"
        log_warn "Skipping inference — run Step 5 (training) first"
        return
    fi

    log_info "Using checkpoint: $CKPT"

    # Find a sample input structure
    SAMPLE_PDB=""
    for candidate in \
        "$PDB_DATA_DIR/paired/"*.pdb \
        "$PDB_DATA_DIR/xray/"*.pdb \
        "$PDB_DATA_DIR/nmr/"*.pdb; do
        if [ -f "$candidate" ]; then
            SAMPLE_PDB="$candidate"
            break
        fi
    done

    if [ -z "$SAMPLE_PDB" ]; then
        log_warn "No sample PDB found for inference demo"
        log_info "To predict manually:"
        echo "  python inference/sample.py --checkpoint $CKPT --input your_structure.pdb"
        return
    fi

    if ! confirm_step "Run inference on sample structure: $(basename $SAMPLE_PDB)"; then
        log_warn "Step 6 skipped"
        return
    fi

    show_eta "30 seconds – 2 minutes"
    mkdir -p ./predictions

    python inference/sample.py \
        --checkpoint   "$CKPT" \
        --input        "$SAMPLE_PDB" \
        --n_conformers "$N_CONFORMERS" \
        --n_steps      20 \
        --method       heun \
        --output       "./predictions/$(basename ${SAMPLE_PDB%.pdb})"

    log_success "Ensemble written to ./predictions/"
    step_done "step6"
}

# ============================================================
#  STEP 7 — EVALUATE ON HELD-OUT SET
# ============================================================

run_step7_evaluate() {
    log_step "STEP 7: Evaluate on Held-out NMR/X-ray Pairs"

    if step_already_done "step7"; then
        log_info "Step 7 already completed"
        return
    fi

    CKPT=""
    for candidate in \
        "$CHECKPOINT_DIR/ckpt_best.pt" \
        "$CHECKPOINT_DIR/ckpt_final.pt"; do
        [ -f "$candidate" ] && CKPT="$candidate" && break
    done

    if [ -z "$CKPT" ]; then
        log_warn "No checkpoint found — skipping evaluation"
        return
    fi

    PAIRS_JSON="$PDB_DATA_DIR/paired.json"
    if [ ! -f "$PAIRS_JSON" ]; then
        log_warn "Paired dataset JSON not found at $PAIRS_JSON"
        log_warn "Skipping evaluation"
        return
    fi

    if ! confirm_step "Evaluate ConformerFlow on held-out NMR/X-ray pairs"; then
        log_warn "Step 7 skipped"
        return
    fi

    show_eta "10–60 minutes depending on number of proteins"
    mkdir -p ./validation_results

    python evaluation/validate.py \
        --checkpoint  "$CKPT" \
        --pairs_json  "$PAIRS_JSON" \
        --xray_dir    "$PDB_DATA_DIR/paired" \
        --nmr_dir     "$PDB_DATA_DIR/paired" \
        --output_dir  "./validation_results" \
        --n_conformers "$N_CONFORMERS" \
        --n_steps     20 \
        --method      heun

    log_success "Validation results saved to ./validation_results/"
    step_done "step7"
}

# ============================================================
#  MAIN — Interactive menu + sequential execution
# ============================================================

main() {
    clear
    log_banner

    echo -e "${BOLD}Working directory:${NC} $(pwd)"
    echo -e "${BOLD}Progress log:${NC}     $STEP_LOG"
    echo ""

    # Show progress from previous runs
    if [ -f "$STEP_LOG" ]; then
        echo -e "${GREEN}Previously completed steps:${NC}"
        cat "$STEP_LOG" | sed 's/step/  ✓  Step /g'
        echo ""
    fi

    echo -e "${BOLD}What would you like to do?${NC}"
    echo ""
    echo "  [A] Run ALL steps sequentially (recommended for first run)"
    echo "  [0] Step 0 — Install dependencies"
    echo "  [1] Step 1 — Fetch PDB data from RCSB"
    echo "  [2] Step 2 — Parse NMR structures"
    echo "  [3] Step 3 — Parse X-ray structures"
    echo "  [4] Step 4 — Filter & split dataset"
    echo "  [5] Step 5 — Train the model"
    echo "  [6] Step 6 — Generate ensemble (inference)"
    echo "  [7] Step 7 — Evaluate on held-out set"
    echo "  [R] Reset progress log (re-run completed steps)"
    echo "  [Q] Quit"
    echo ""

    read -r -p "Choice: " choice

    case "$choice" in
        [Aa])
            check_file_structure
            run_step0_environment
            run_step1_fetch
            run_step2_parse_nmr
            run_step3_parse_xray
            run_step4_filter
            run_step5_train
            run_step6_inference
            run_step7_evaluate

            echo ""
            echo -e "${BOLD}${GREEN}"
            echo "  ╔══════════════════════════════════════════╗"
            echo "  ║     ConformerFlow Pipeline Complete!     ║"
            echo "  ╚══════════════════════════════════════════╝"
            echo -e "${NC}"
            ;;
        0) check_file_structure; run_step0_environment ;;
        1) check_file_structure; run_step1_fetch ;;
        2) check_file_structure; run_step2_parse_nmr ;;
        3) check_file_structure; run_step3_parse_xray ;;
        4) check_file_structure; run_step4_filter ;;
        5) check_file_structure; run_step5_train ;;
        6) check_file_structure; run_step6_inference ;;
        7) check_file_structure; run_step7_evaluate ;;
        [Rr])
            rm -f "$STEP_LOG"
            log_success "Progress log reset — all steps will re-run"
            ;;
        [Qq]) echo "Goodbye."; exit 0 ;;
        *)    log_warn "Unknown option: $choice"; main ;;
    esac
}

main
