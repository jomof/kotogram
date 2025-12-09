#!/bin/bash
# Train formality classifier on full Japanese sentence corpus
#
# Usage:
#   ./train_formality.sh                    # Default training
#   ./train_formality.sh --pretrain-mlm     # With MLM pretraining
#   ./train_formality.sh --encoder bilstm   # Use BiLSTM instead of Transformer

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
        python -m pip install torch
    fi

    if ! python -c "import sudachipy" 2>/dev/null; then
        echo "SudachiPy not found. Installing..."
        python -m pip install sudachipy sudachidict_core
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
DATA_PATH="data/jpn_sentences.tsv"
OUTPUT_DIR="models/formality"
EPOCHS=20
BATCH_SIZE=64
ENCODER="transformer"
EMBED_DIM=128
HIDDEN_DIM=256
NUM_LAYERS=3
PRETRAIN_MLM=""
PRETRAIN_EPOCHS=5
MAX_SAMPLES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
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
        --encoder)
            ENCODER="$2"
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
        --help)
            echo "Train formality classifier on Japanese sentence corpus"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data PATH           Path to TSV file (default: data/jpn_sentences.tsv)"
            echo "  --output DIR          Output directory (default: models/formality)"
            echo "  --epochs N            Training epochs (default: 20)"
            echo "  --batch-size N        Batch size (default: 64)"
            echo "  --encoder TYPE        Encoder type: transformer or bilstm (default: transformer)"
            echo "  --embed-dim N         Embedding dimension (default: 128)"
            echo "  --hidden-dim N        Hidden layer dimension (default: 256)"
            echo "  --num-layers N        Number of encoder layers (default: 3)"
            echo "  --pretrain-mlm        Enable masked LM pretraining"
            echo "  --pretrain-epochs N   MLM pretraining epochs (default: 5)"
            echo "  --max-samples N       Limit samples (for testing)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Formality Classifier Training"
echo "=============================================="
echo "Data:           $DATA_PATH"
echo "Output:         $OUTPUT_DIR"
echo "Epochs:         $EPOCHS"
echo "Batch size:     $BATCH_SIZE"
echo "Encoder:        $ENCODER"
echo "Embedding dim:  $EMBED_DIM"
echo "Hidden dim:     $HIDDEN_DIM"
echo "Num layers:     $NUM_LAYERS"
if [ -n "$PRETRAIN_MLM" ]; then
    echo "MLM pretrain:   $PRETRAIN_EPOCHS epochs"
fi
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples:    ${MAX_SAMPLES#--max-samples }"
fi
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python -m kotogram.formality_classifier \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --encoder "$ENCODER" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    $PRETRAIN_MLM \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    $MAX_SAMPLES \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log:   $OUTPUT_DIR/training.log"
echo "=============================================="
