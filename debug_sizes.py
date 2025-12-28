import glob
import os


def check_consistency(dir_path):
    print(f"Checking {dir_path}...")
    offsets_path = os.path.join(dir_path, "offsets.bin")
    if not os.path.exists(offsets_path):
        print("  No offsets.bin")
        return

    off_size = os.path.getsize(offsets_path)
    num_samples = (off_size // 4) - 1
    print(f"  Offsets size: {off_size} bytes -> {num_samples} samples")

    labels_pat = os.path.join(dir_path, "*.labels.bin_*")
    # Also check merged
    labels_pat2 = os.path.join(dir_path, "labels.bin_*")

    files = glob.glob(labels_pat) + glob.glob(labels_pat2)
    for f in files:
        sz = os.path.getsize(f)
        fname = os.path.basename(f)

        # Determine expected size based on type
        # f_val, g_val: float32 (4 bytes) -> sz // 4 should match num_samples
        # f_prag, g_prag, gram: uint8 (1 byte) -> sz should match num_samples
        # reg_ids: ragged (skip)

        if "reg_ids" in fname:
            continue

        expected = num_samples
        itemsize = 1
        if "_val" in fname:
            expected = num_samples * 4
            itemsize = 4

        if sz != expected:
            print(f"  MISMATCH: {fname} size {sz} (expected {expected})")
            diff = sz - expected
            print(f"    Diff: {diff} bytes ({diff // itemsize} items)")
        else:
            print(f"  MATCH: {fname}")


def main():
    check_consistency(".cache/style_dataset")
    check_consistency("models/style")


if __name__ == "__main__":
    main()
