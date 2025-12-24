#!/bin/bash
# Train style classifier (formality + gender + grammaticality) on full Japanese sentence corpus
#
# Usage:
#   ./train_style.sh                    # Default training
#   ./train_style.sh --pretrain-mlm     # With MLM pretraining

set -e
set -o pipefail

# Resolve TRAIN_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$TRAIN_ROOT" ]; then
    TRAIN_ROOT="."
elif [[ "$TRAIN_ROOT" != /* ]]; then
    # If relative, make it relative to the script directory
    TRAIN_ROOT="$SCRIPT_DIR/$TRAIN_ROOT"
fi
export TRAIN_ROOT


# Setup virtual environment and dependencies
setup_environment() {
    # VENV always stays with the script/project, not the TRAIN_ROOT
    VENV_DIR="$SCRIPT_DIR/.venv"

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"

    echo "Checking dependencies..."

    if ! python -c "import torch" 2>/dev/null; then
        echo "PyTorch not found. Installing..."
        python -m pip install --upgrade pip
        python -m pip install torch numpy
    fi

    if ! python -c "import numpy" 2>/dev/null; then
        echo "NumPy not found. Installing..."
        python -m pip install numpy
    fi

    if ! python -c "import sudachidict_full" 2>/dev/null; then
        echo "SudachiPy/dictionary not found. Installing..."
        python -m pip install sudachipy sudachidict_full
    fi

    if ! python -c "import rich" 2>/dev/null; then
        echo "Rich library not found. Installing..."
        python -m pip install rich
    fi

    # Check if kotogram is installed in the environment (ignoring CWD)
    if ! python -c "import sys; sys.path = [p for p in sys.path if p != '']; import kotogram" 2>/dev/null; then
        echo "Kotogram not found in site-packages. Installing from current directory..."
        python -m pip install -e "$SCRIPT_DIR"
    fi

    echo "Dependencies OK (using venv: $VENV_DIR)"
    echo ""
}

setup_environment

# Default configuration
DATA_DIR=$(python3 -m scripts.locations data)
MODELS_DIR=$(python3 -m scripts.locations models)

DATA_PATH="$DATA_DIR/jpn_sentences*.tsv"  # Filtered to exclude known errors
AGRAMMATIC_SENTENCES_PATH=""
AGRAMMATIC_PATTERN="$DATA_DIR/jpn_agrammatic*.tsv"
OUTPUT_DIR="$MODELS_DIR/style"
EPOCHS=""
# Batch settings
MICRO_BATCH_SIZE=32       # Batch size per device
TARGET_GLOBAL_BATCH_SIZE=128 # Operations per optimizer step (32 * 4 GPUs = 128)
EMBED_DIM=256
HIDDEN_DIM=512
NUM_LAYERS=3
NUM_HEADS=8
PRETRAIN_MLM=""
PRETRAIN_EPOCHS=5
ENCODER_LR_FACTOR=0.1
LEARNING_RATE=1e-4
FORMALITY_WEIGHT=1.0
GENDER_WEIGHT=1.0
GRAMMATICALITY_WEIGHT=5.0
FP16=""
FP8=""
RESUME=""
RETRAIN=""
CONFUSION=""
LABEL_ONLY=""

PERCENT=""

# KC Training Defaults
PRETRAIN_KC=""
KC_EPOCHS=3
KC_K=1024
KC_TOPK=8
KC_FREEZE_ENCODER_EPOCHS=1
KC_SPARSITY_WEIGHT=1e-3
KC_TARGET_HEADS="lemma,pos,conjugated_form"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in

        --agrammatic-pattern)
            AGRAMMATIC_PATTERN="$2"
            shift 2
            ;;
        --no-agrammatic-data)
            AGRAMMATIC_PATTERN=""
            shift
            ;;

        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --embed-dim)
            EMBED_DIM="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --pretrain-mlm)
            PRETRAIN_MLM="--pretrain-mlm"
            shift
            ;;
        --pretrain-kc)
            PRETRAIN_KC="--pretrain-kc"
            shift
            ;;
        --kc-epochs)
            KC_EPOCHS="$2"
            shift 2
            ;;
        --kc-k)
            KC_K="$2"
            shift 2
            ;;
        --kc-topk)
            KC_TOPK="$2"
            shift 2
            ;;
        --kc-freeze-encoder-epochs)
            KC_FREEZE_ENCODER_EPOCHS="$2"
            shift 2
            ;;
        --kc-sparsity-weight)
            KC_SPARSITY_WEIGHT="$2"
            shift 2
            ;;
        --kc-target-heads)
            KC_TARGET_HEADS="$2"
            shift 2
            ;;
        --force-relabel)
            FORCE_RELABEL="--force-relabel"
            shift
            ;;
        --pretrain-epochs)
            PRETRAIN_EPOCHS="$2"
            shift 2
            ;;
        --encoder-lr-factor)
            ENCODER_LR_FACTOR="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --formality-weight)
            FORMALITY_WEIGHT="$2"
            shift 2
            ;;
        --gender-weight)
            GENDER_WEIGHT="$2"
            shift 2
            ;;
        --grammaticality-weight)
            GRAMMATICALITY_WEIGHT="$2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --fp8)
            FP8="--fp8"
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --retrain)
            RETRAIN="--retrain"
            shift
            ;;
        --confusion)
            CONFUSION="--confusion"
            shift
            ;;
        --label)
            LABEL_ONLY=1
            shift
            ;;

        --percent)
            PERCENT="--percent $2"
            shift 2
            ;;
        --help)
            echo "Train style classifier (formality + gender + grammaticality) on Japanese sentence corpus"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""

            echo ""
            echo "Training Options:"
            echo "  --epochs N            Training epochs (default: 20)"
            echo "  --batch-size N        Batch size (default: 64)"
            echo "  --learning-rate F     Base learning rate (default: 1e-4)"
            echo "  --pretrain-mlm        Enable masked LM pretraining"
            echo "  --pretrain-epochs N   MLM pretraining epochs (default: 5)"
            echo "  --encoder-lr-factor F LR factor for encoder in fine-tuning (default: 0.1)"
            echo ""
            echo "Data Options:"
            echo "  --agrammatic-pattern PATTERN Pattern for agrammatic TSV files (default: $TRAIN_ROOT/data/jpn_agrammatic*.tsv)"
            echo "  --no-agrammatic-data  Disable loading agrammatic data file"
            echo ""
            echo "Multi-task Loss Weights:"
            echo "  --formality-weight F  Weight for formality loss (default: 1.0)"
            echo "  --gender-weight F     Weight for gender loss (default: 1.0)"
            echo "  --grammaticality-weight F Weight for grammaticality loss (default: 1.0)"
            echo ""
            echo "Model Architecture:"
            echo "  --embed-dim N         Model dimension (default: 256)"
            echo "  --hidden-dim N        Hidden layer dimension (default: 512)"
            echo "  --num-layers N        Number of encoder layers (default: 3)"
            echo "  --num-heads N         Number of attention heads (default: 8)"
            echo "  --fp16                Save model in float16 (half size, minimal accuracy loss)"
            echo "  --fp8                 Save model in float8 (quarter size, requires PyTorch 2.1+)"
            echo "  --resume              Resume training from checkpoint in output directory"
            echo "  --retrain             Retrain from scratch using parameters from checkpoint"
            echo "  --confusion           Print confusion matrices for existing model and exit"
            echo "  --label               Run ONLY the labeling/preprocessing phase and exit"

            echo ""
            echo "  --percent N           Percentage of data to use (1-100)"
            echo ""
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================"
echo "Style Classifier Training (Formality + Gender + Grammaticality)"
echo "========================================================"
echo "Data:           $DATA_PATH"
if [ -n "$AGRAMMATIC_SENTENCES_PATH" ]; then
    echo "Agrammatic sentences: $AGRAMMATIC_SENTENCES_PATH"
fi
if [ -n "$AGRAMMATIC_PATTERN" ]; then
    echo "Agrammatic pattern: $AGRAMMATIC_PATTERN"
fi
echo "Output:         $OUTPUT_DIR"
if [ -n "$EPOCHS" ]; then
    echo "Epochs:         $EPOCHS"
else
    echo "Epochs:         (default or restored from checkpoint)"
fi
echo "Batch size:     $BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Model dim:      $EMBED_DIM"
echo "Hidden dim:     $HIDDEN_DIM"
echo "Num layers:     $NUM_LAYERS"
echo "Num heads:      $NUM_HEADS"
echo "Formality wt:   $FORMALITY_WEIGHT"
echo "Gender wt:      $GENDER_WEIGHT"
echo "Grammatic wt:   $GRAMMATICALITY_WEIGHT"
if [ -n "$PRETRAIN_MLM" ]; then
    echo "MLM pretrain:   $PRETRAIN_EPOCHS epochs"
    echo "Encoder LR:     ${ENCODER_LR_FACTOR}x base LR during fine-tuning"
fi
if [ -n "$FP8" ]; then
    echo "Precision:      float8 (quarter size)"
elif [ -n "$FP16" ]; then
    echo "Precision:      float16 (half size)"
fi
if [ -n "$RESUME" ]; then
    echo "Resume:         from checkpoint"
fi
if [ -n "$RETRAIN" ]; then
    echo "Retrain:        from scratch using parameters from checkpoint"
fi
if [ -n "$CONFUSION" ]; then
    echo "Action:         Print confusion matrices (no training)"
fi
if [ -n "$LABEL_ONLY" ]; then
    echo "Action:         Preprocessing/Labeling only"
fi


if [ -n "$PERCENT" ]; then
    echo "Data usage:     ${PERCENT#--percent }%"
fi
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Combined output files in cache
CACHE_DIR=$(python3 -m scripts.locations cache)
SUPPORT_DIR=$(python3 -m scripts.locations style-support)

COMBINED_GRAM_FILE="$CACHE_DIR/grammatic_combined.tsv"
COMBINED_AGRAM_FILE="$CACHE_DIR/agrammatic_combined.tsv"

mkdir -p "$CACHE_DIR"
mkdir -p "$SUPPORT_DIR"

# Store patterns for later use
GRAM_DATA_PATTERN="$DATA_PATH"
# Note: we use the combined files for the train_style.py call
DATA_PATH="$COMBINED_GRAM_FILE"
AGRAMMATIC_DATA_PATH="$COMBINED_AGRAM_FILE"

# Enable MPS fallback for unsupported ops (Mac Apple Silicon)
# Enable MPS fallback for unsupported ops (Mac Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Silence OMP warnings and prevent thread oversubscription
export OMP_NUM_THREADS=1

# Detect Environment
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" -gt 0 ] && [ -n "$DEBUG" ]; then
    echo "Detected GPUs: $NUM_GPUS"
fi

# Calculate devices for math alignment
# If 0 GPUs, we assume 1 device (MPS or CPU)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_DEVICES=1
else
    NUM_DEVICES=$NUM_GPUS
fi

# Calculate workers (for DataLoader/Preprocessing)
if [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_COUNT=$(sysctl -n hw.ncpu)
else
    CPU_COUNT=$(nproc 2>/dev/null || echo "4")
fi
NUM_WORKERS=$((CPU_COUNT > 1 ? CPU_COUNT - 1 : 1))

# Calculate Gradient Accumulation to match Target Global Batch
# Formula: GradAccum = Target / (Devices * MicroBatch)
TOTAL_CURRENT_BATCH=$((NUM_DEVICES * MICRO_BATCH_SIZE))

# Gradient Accumulation check
if [ "$TOTAL_CURRENT_BATCH" -lt "$TARGET_GLOBAL_BATCH_SIZE" ]; then
    GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / TOTAL_CURRENT_BATCH))
else
    GRAD_ACCUM_STEPS=1 # No accumulation needed if micro batch >= target global batch
fi

# Ensure at least 1 step
if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi

# Set defaults if not provided
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$MICRO_BATCH_SIZE
fi

# Defaults for precision are handled in scripts/train_style.py

# Build command definition
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS"
else
    LAUNCHER="python"
fi

# Configuration Summary
if [ -n "$DEBUG" ]; then
    echo "Configuration: $LAUNCHER, Batch: $BATCH_SIZE, Accum: $GRAD_ACCUM_STEPS, FP16: ${FP16:-off}"
fi

# Build command
CMD="$LAUNCHER -m scripts.train_style \
    --batch-size $BATCH_SIZE \
    --embed-dim $EMBED_DIM \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --learning-rate $LEARNING_RATE \
    --pretrain-epochs $PRETRAIN_EPOCHS \
    --encoder-lr-factor $ENCODER_LR_FACTOR \
    --gender-weight $GENDER_WEIGHT \
    --grammaticality-weight $GRAMMATICALITY_WEIGHT"

if [ -n "$PRETRAIN_MLM" ]; then
    CMD="$CMD --pretrain-mlm"
fi

if [ -n "$PRETRAIN_KC" ]; then
    CMD="$CMD --pretrain-kc"
fi

CMD="$CMD --kc-epochs $KC_EPOCHS \
    --kc-k $KC_K \
    --kc-topk $KC_TOPK \
    --kc-freeze-encoder-epochs $KC_FREEZE_ENCODER_EPOCHS \
    --kc-sparsity-weight $KC_SPARSITY_WEIGHT \
    --kc-target-heads \"$KC_TARGET_HEADS\""





if [ -n "$FP8" ]; then
    CMD="$CMD --fp8"
elif [ -n "$FP16" ]; then
    CMD="$CMD --fp16"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume"
fi

if [ -n "$RETRAIN" ]; then
    CMD="$CMD --retrain"
fi



if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [ -n "$PERCENT" ]; then
    CMD="$CMD $PERCENT"
fi

# Add grad accum steps 
CMD="$CMD --grad-accum-steps $GRAD_ACCUM_STEPS"

# Run Preprocessing Phase (Single Process)
# This ensures that Kotogram parsing and dataset caching happens once, cleanly,
# before launching training (whether single or multi-process).
echo "=============================================="
echo "Running Preprocessing Phase..."
echo "=============================================="
# Construct preprocessing command (always use python, single process)
    # Construct labeling command
    PREPROC_CMD="python -m scripts.label --grammatic-pattern \"$GRAM_DATA_PATTERN\" $FORCE_RELABEL"

    if [ -n "$AGRAMMATIC_SENTENCES_PATH" ] || [ -n "$AGRAMMATIC_PATTERN" ]; then 
        PREPROC_CMD="$PREPROC_CMD --agrammatic-pattern \"$AGRAMMATIC_PATTERN\""
    fi

if [ -n "$DEBUG" ]; then
    echo "Command: $PREPROC_CMD"
else
    echo "Executing preprocessing script..."
fi
eval $PREPROC_CMD || exit 1

# Update line counts for log display
echo "Combined grammatic data: $DATA_PATH ($(wc -l <"$DATA_PATH" | xargs) lines)"
if [ -f "$AGRAMMATIC_DATA_PATH" ]; then
    echo "Combined agrammatic data: $AGRAMMATIC_DATA_PATH ($(wc -l <"$AGRAMMATIC_DATA_PATH" | xargs) lines)"
fi
echo "Preprocessing complete."

if [ -n "$LABEL_ONLY" ]; then
    echo "Labeling only requested. Exiting."
    exit 0
fi
echo ""

# Confusion evaluation (runs after preprocessing)
if [ -n "$CONFUSION" ]; then
    echo "=============================================="
    echo "Running Confusion Matrix Evaluation..."
    echo "=============================================="
    python -m scripts.confusion \
        ${PERCENT:+--percent ${PERCENT#--percent }}
    exit 0
fi


# Run training
if [ -n "$DEBUG" ]; then
    echo "Command: $CMD"
else
    echo "Starting training run..."
fi
eval $CMD 2>&1 | tee "$SUPPORT_DIR/training.log"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log:   $SUPPORT_DIR/training.log"
echo "=============================================="
echo ""
echo "Generating confusion report..."
python -m scripts.confusion \
    ${PERCENT:+--percent ${PERCENT#--percent }}
