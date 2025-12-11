#!/bin/bash
# Train style classifier (formality + gender + grammaticality) on full Japanese sentence corpus
#
# Usage:
#   ./train_style.sh                    # Default training
#   ./train_style.sh --pretrain-mlm     # With MLM pretraining

set -e

# Setup virtual environment and dependencies
setup_environment() {
    VENV_DIR=".venv"

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

    if ! python -c "import kotogram" 2>/dev/null; then
        echo "Kotogram not found. Installing from current directory..."
        python -m pip install -e .
    fi

    echo "Dependencies OK (using venv: $VENV_DIR)"
    echo ""
}

setup_environment

# Default configuration
DATA_PATH="data/jpn_sentences.tsv"  # Filtered to exclude known errors
EXTRA_DATA_PATH="data/unpragmatic_sentences.tsv"
AGRAMMATIC_DATA_PATH="data/jpn_agrammatic.tsv"
OUTPUT_DIR="models/style"
EPOCHS=20
BATCH_SIZE=64
EMBED_DIM=256
HIDDEN_DIM=512
NUM_LAYERS=3
NUM_HEADS=8
PRETRAIN_MLM=""
PRETRAIN_EPOCHS=5
MAX_SAMPLES=""
ENCODER_LR_FACTOR=0.1
LEARNING_RATE=1e-4
FORMALITY_WEIGHT=1.0
GENDER_WEIGHT=1.0
GRAMMATICALITY_WEIGHT=1.0
FP16=""
FP8=""
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --extra-data)
            EXTRA_DATA_PATH="$2"
            shift 2
            ;;
        --no-extra-data)
            EXTRA_DATA_PATH=""
            shift
            ;;
        --agrammatic-data)
            AGRAMMATIC_DATA_PATH="$2"
            shift 2
            ;;
        --no-agrammatic-data)
            AGRAMMATIC_DATA_PATH=""
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
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
        --pretrain-epochs)
            PRETRAIN_EPOCHS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
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
        --help)
            echo "Train style classifier (formality + gender + grammaticality) on Japanese sentence corpus"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Data Options:"
            echo "  --data PATH           Path to primary TSV file (default: data/jpn_sentences.tsv)"
            echo "  --extra-data PATH     Path to extra TSV file (default: data/unpragmatic_sentences.tsv)"
            echo "  --no-extra-data       Disable loading extra data file"
            echo "  --agrammatic-data PATH Path to agrammatic TSV file (default: data/jpn_agrammatic.tsv)"
            echo "  --no-agrammatic-data  Disable loading agrammatic data file"
            echo "  --output DIR          Output directory (default: models/style)"
            echo "  --max-samples N       Limit samples (for testing)"
            echo ""
            echo "Training Options:"
            echo "  --epochs N            Training epochs (default: 20)"
            echo "  --batch-size N        Batch size (default: 64)"
            echo "  --learning-rate F     Base learning rate (default: 1e-4)"
            echo "  --pretrain-mlm        Enable masked LM pretraining"
            echo "  --pretrain-epochs N   MLM pretraining epochs (default: 5)"
            echo "  --encoder-lr-factor F LR factor for encoder in fine-tuning (default: 0.1)"
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
if [ -n "$EXTRA_DATA_PATH" ]; then
    echo "Extra data:     $EXTRA_DATA_PATH"
fi
if [ -n "$AGRAMMATIC_DATA_PATH" ]; then
    echo "Agrammatic:     $AGRAMMATIC_DATA_PATH"
fi
echo "Output:         $OUTPUT_DIR"
echo "Epochs:         $EPOCHS"
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
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples:    ${MAX_SAMPLES#--max-samples }"
fi
if [ -n "$FP8" ]; then
    echo "Precision:      float8 (quarter size)"
elif [ -n "$FP16" ]; then
    echo "Precision:      float16 (half size)"
fi
if [ -n "$RESUME" ]; then
    echo "Resume:         from checkpoint"
fi
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Enable MPS fallback for unsupported ops (Mac Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Build command
CMD="python -m kotogram.style_classifier \
    --data \"$DATA_PATH\" \
    --output \"$OUTPUT_DIR\" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --embed-dim $EMBED_DIM \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --learning-rate $LEARNING_RATE \
    --pretrain-epochs $PRETRAIN_EPOCHS \
    --encoder-lr-factor $ENCODER_LR_FACTOR \
    --formality-weight $FORMALITY_WEIGHT \
    --gender-weight $GENDER_WEIGHT \
    --grammaticality-weight $GRAMMATICALITY_WEIGHT"

if [ -n "$PRETRAIN_MLM" ]; then
    CMD="$CMD --pretrain-mlm"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD $MAX_SAMPLES"
fi

if [ -n "$EXTRA_DATA_PATH" ]; then
    CMD="$CMD --extra-data \"$EXTRA_DATA_PATH\""
fi

if [ -n "$AGRAMMATIC_DATA_PATH" ]; then
    CMD="$CMD --agrammatic-data \"$AGRAMMATIC_DATA_PATH\""
fi

if [ -n "$FP8" ]; then
    CMD="$CMD --fp8"
elif [ -n "$FP16" ]; then
    CMD="$CMD --fp16"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume"
fi

# Run training
eval $CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log:   $OUTPUT_DIR/training.log"
echo "=============================================="
