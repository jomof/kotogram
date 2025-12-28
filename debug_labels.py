import glob
from collections import Counter


def check_file(path):
    print(f"Checking {path}...")
    with open(path, "rb") as f:
        data = f.read()

    # Assuming uint8 (B)
    vals = list(data)
    c = Counter(vals)
    print(f"  Unique values: {c}")
    return c


def main():
    # Check .cache shards
    patterns = [
        ".cache/style_dataset/shard_*.labels.bin_f_prag",
        ".cache/style_dataset/labels.bin_f_prag",
        "models/style/labels.bin_f_prag",
    ]

    for pat in patterns:
        files = glob.glob(pat)
        if not files:
            print(f"No files found for {pat}")
            continue

        for f in files:
            check_file(f)


if __name__ == "__main__":
    main()
