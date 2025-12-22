
import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to model file (optionally fp8).")
    parser.add_argument("--checkpoint", default="models/style/checkpoint.pt", help="Path to input checkpoint")
    parser.add_argument("--output", default="models/style/model.pt", help="Path to output model file")
    parser.add_argument("--fp8", action="store_true", help="Convert to fp8")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    # Load to CPU to avoid CUDA OOM or compatibility issues
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
         # Fallback for older pytorch versions without weights_only
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Found model_state_dict in checkpoint.")
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume the file itself is the state dict if not a checkpoint wrapper
        print("Checkpoint structure not found; assuming direct state dict.")
        state_dict = checkpoint
    
    if args.fp8:
        if not hasattr(torch, "float8_e4m3fn"):
            print("Error: FP8 requires PyTorch 2.1+ and support for float8_e4m3fn")
            sys.exit(1)
            
        print("Converting to FP8...")
        # Convert float32 weights to float8
        new_state_dict = {}
        for k, v in state_dict.items():
            if v.dtype == torch.float32:
                new_state_dict[k] = v.to(torch.float8_e4m3fn)
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving model to {args.output}...")
    torch.save(state_dict, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
