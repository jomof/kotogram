#!/bin/bash
# Train style classifier (formality + gender + grammaticality) on full Japanese sentence corpus
#
# Usage:
#   ./train_style.sh                    # Default training
#   ./train_style.sh --pretrain-mlm     # With MLM pretraining

set -e
set -o pipefail

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

    if ! python -c "import rich" 2>/dev/null; then
        echo "Rich library not found. Installing..."
        python -m pip install rich
    fi

    # Check if kotogram is installed in the environment (ignoring CWD)
    if ! python -c "import sys; sys.path = [p for p in sys.path if p != '']; import kotogram" 2>/dev/null; then
        echo "Kotogram not found in site-packages. Installing from current directory..."
        python -m pip install -e .
    fi

    echo "Dependencies OK (using venv: $VENV_DIR)"
    echo ""
}

setup_environment

# Default configuration
DATA_PATH="data/jpn_sentences*.tsv"  # Filtered to exclude known errors
EXTRA_DATA_PATH=""
AGRAMMATIC_PATTERN="data/jpn_agrammatic*.tsv"
OUTPUT_DIR="models/style"
EPOCHS=20
OUTPUT_DIR="models/style"
EPOCHS=20
# Batch settings
MICRO_BATCH_SIZE=32       # Batch size per device
TARGET_GLOBAL_BATCH_SIZE=128 # Operations per optimizer step (32 * 4 GPUs = 128)
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
GRAMMATICALITY_WEIGHT=5.0
FP16=""
FP8=""
RESUME=""
RETRAIN=""
CONFUSION=""

EXCLUDE_FEATURES=""
PERCENT=""

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
        --agrammatic-pattern)
            AGRAMMATIC_PATTERN="$2"
            shift 2
            ;;
        --no-agrammatic-data)
            AGRAMMATIC_PATTERN=""
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
        --retrain)
            RETRAIN="--retrain"
            shift
            ;;
        --confusion)
            CONFUSION="--confusion"
            shift
            ;;
        --exclude-features)
            EXCLUDE_FEATURES="$2"
            shift 2
            ;;
        --percent)
            PERCENT="--percent $2"
            shift 2
            ;;
        --test)
            IS_TEST=1
            OUTPUT_DIR="models/test_style"
            # Set defaults for test mode if not already specified (simple approach: just set them)
            # Users can override by passing --epochs N after --test if they really want
            EPOCHS=1
            MAX_SAMPLES="--max-samples 100"
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
            echo "  --agrammatic-pattern PATTERN Pattern for agrammatic TSV files (default: data/jpn_agrammatic*.tsv)"
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
            echo "  --retrain             Retrain from scratch using parameters from checkpoint"
            echo "  --confusion           Print confusion matrices for existing model and exit"
            echo ""
            echo "Feature Ablation:"
            echo "  --exclude-features F  Comma-separated features to exclude (for ablation study)"
            echo "                        Valid: surface,pos,pos_detail1,pos_detail2,conjugated_type,conjugated_form,lemma"

            echo ""
            echo "  --percent N           Percentage of data to use (1-100)"
            echo ""
            echo "  --test                Run in test mode (output to models/test_style, 1 epoch, 100 samples)"
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
if [ -n "$AGRAMMATIC_PATTERN" ]; then
    echo "Agrammatic pattern: $AGRAMMATIC_PATTERN"
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
if [ -n "$RETRAIN" ]; then
    echo "Retrain:        from scratch using parameters from checkpoint"
fi
if [ -n "$CONFUSION" ]; then
    echo "Action:         Print confusion matrices (no training)"
fi


if [ -n "$PERCENT" ]; then
    echo "Data usage:     ${PERCENT#--percent }%"
fi
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process grammatical data (from pattern)
if ls $DATA_PATH 1> /dev/null 2>&1; then
    mkdir -p .cache
    
    COMBINED_GRAM_FILE=".cache/grammatic_combined.tsv"
    TEMP_GRAM_FILE="${COMBINED_GRAM_FILE}.tmp"
    
    # Combine and deduplicate by 3rd column (sentence) called from pattern
    awk -F'\t' '!seen[$3]++' $DATA_PATH > "$TEMP_GRAM_FILE"
    
    # Only update if content changed
    if [ -f "$COMBINED_GRAM_FILE" ] && cmp -s "$TEMP_GRAM_FILE" "$COMBINED_GRAM_FILE"; then
        echo "Grammatic data unchanged, using cached file."
        rm "$TEMP_GRAM_FILE"
    else
        echo "Updating combined grammatic data..."
        mv "$TEMP_GRAM_FILE" "$COMBINED_GRAM_FILE"
    fi
    
    DATA_PATH="$COMBINED_GRAM_FILE"
    echo "Combined grammatic data: $DATA_PATH ($(wc -l < "$DATA_PATH" | xargs) lines)"
else
    echo "Error: No files matched grammatic data pattern: $DATA_PATH"
    exit 1
fi

# Process agrammatic data
AGRAMMATIC_DATA_PATH=""
if [ -n "$AGRAMMATIC_PATTERN" ]; then
    # Check if any files match the pattern
    if ls $AGRAMMATIC_PATTERN 1> /dev/null 2>&1; then
        echo "Processing agrammatic data from: $AGRAMMATIC_PATTERN"
        mkdir -p .cache
        
        COMBINED_FILE=".cache/agrammatic_combined.tsv"
        TEMP_FILE="${COMBINED_FILE}.tmp"
        
        # Combine and deduplicate by 3rd column (sentence), keeping the first occurrence
        # We assume files are TSV and 3rd column is the sentence
        awk -F'\t' '!seen[$3]++' $AGRAMMATIC_PATTERN > "$TEMP_FILE"
        
        # Only update if content has changed (preserves mtime for caching)
        if [ -f "$COMBINED_FILE" ] && cmp -s "$TEMP_FILE" "$COMBINED_FILE"; then
            echo "Agrammatic data unchanged, using cached file."
            rm "$TEMP_FILE"
        else
            echo "Updating combined agrammatic data..."
            mv "$TEMP_FILE" "$COMBINED_FILE"
        fi
        
        AGRAMMATIC_DATA_PATH="$COMBINED_FILE"
        echo "Combined agrammatic data: $AGRAMMATIC_DATA_PATH ($(wc -l < "$AGRAMMATIC_DATA_PATH" | xargs) lines)"
    else
        echo "Warning: No files matched agrammatic pattern: $AGRAMMATIC_PATTERN"
    fi
fi

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

echo "Auto-config: Target Global Batch=$TARGET_GLOBAL_BATCH_SIZE"
echo "             Devices=$NUM_DEVICES, MicroBatch=$MICRO_BATCH_SIZE"
echo "             => GradAccumSteps=$GRAD_ACCUM_STEPS"


# Set defaults if not provided
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$MICRO_BATCH_SIZE
fi

# Auto-enable FP16 globally for alignment (unless FP8 or user specified otherwise)
if [ -z "$FP16" ] && [ -z "$FP8" ]; then
    FP16="--fp16"
fi

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
CMD="$LAUNCHER scripts/train_style.py \
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

if [ -n "$RETRAIN" ]; then
    CMD="$CMD --retrain"
fi

if [ -n "$CONFUSION" ]; then
    echo "=============================================="
    echo "Running Confusion Matrix Evaluation..."
    echo "=============================================="
    python scripts/confusion.py \
        --output "$OUTPUT_DIR" \
        --data "$DATA_PATH" \
        ${EXTRA_DATA_PATH:+--extra-data "$EXTRA_DATA_PATH"} \
        ${AGRAMMATIC_DATA_PATH:+--agrammatic-data "$AGRAMMATIC_DATA_PATH"} \
        ${PERCENT:+--percent ${PERCENT#--percent }} \
        ${MAX_SAMPLES:+--max-samples ${MAX_SAMPLES#--max-samples }}
    exit 0
fi

if [ -n "$EXCLUDE_FEATURES" ]; then
    CMD="$CMD --exclude-features \"$EXCLUDE_FEATURES\""
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
    PREPROC_CMD="python scripts/train_style.py \
        --data \"$DATA_PATH\" \
        --output \"$OUTPUT_DIR\" \
        --epochs 1 \
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
        --grammaticality-weight $GRAMMATICALITY_WEIGHT \
        --preprocess-only"

    # Append all optional flags
    if [ -n "$PRETRAIN_MLM" ]; then PREPROC_CMD="$PREPROC_CMD --pretrain-mlm"; fi
    if [ -n "$MAX_SAMPLES" ]; then PREPROC_CMD="$PREPROC_CMD $MAX_SAMPLES"; fi
    if [ -n "$EXTRA_DATA_PATH" ]; then PREPROC_CMD="$PREPROC_CMD --extra-data \"$EXTRA_DATA_PATH\""; fi
    if [ -n "$AGRAMMATIC_DATA_PATH" ]; then PREPROC_CMD="$PREPROC_CMD --agrammatic-data \"$AGRAMMATIC_DATA_PATH\""; fi
    if [ -n "$FP8" ]; then PREPROC_CMD="$PREPROC_CMD --fp8";
    elif [ -n "$FP16" ]; then PREPROC_CMD="$PREPROC_CMD --fp16"; fi
    if [ -n "$RESUME" ]; then PREPROC_CMD="$PREPROC_CMD --resume"; fi
    if [ -n "$RETRAIN" ]; then PREPROC_CMD="$PREPROC_CMD --retrain"; fi
    if [ -n "$EXCLUDE_FEATURES" ]; then PREPROC_CMD="$PREPROC_CMD --exclude-features \"$EXCLUDE_FEATURES\""; fi
    if [ -n "$PERCENT" ]; then PREPROC_CMD="$PREPROC_CMD $PERCENT"; fi

if [ -n "$DEBUG" ]; then
    echo "Command: $PREPROC_CMD"
else
    echo "Executing preprocessing script..."
fi
eval $PREPROC_CMD || exit 1
echo "Preprocessing complete."
echo ""

# Run training
if [ -n "$DEBUG" ]; then
    echo "Command: $CMD"
else
    echo "Starting training run..."
fi
eval $CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log:   $OUTPUT_DIR/training.log"
echo "=============================================="
echo ""
echo "Generating confusion report..."
python scripts/confusion.py \
    --output "$OUTPUT_DIR" \
    --data "$DATA_PATH" \
    ${EXTRA_DATA_PATH:+--extra-data "$EXTRA_DATA_PATH"} \
    ${AGRAMMATIC_DATA_PATH:+--agrammatic-data "$AGRAMMATIC_DATA_PATH"} \
    ${PERCENT:+--percent ${PERCENT#--percent }} \
    ${MAX_SAMPLES:+--max-samples ${MAX_SAMPLES#--max-samples }}
