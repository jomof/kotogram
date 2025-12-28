import os
import sys
import tracemalloc


def main() -> None:
    if len(sys.argv) < 3:
        # Just exit, stderr will log usage if we printed to stderr but we print to stdout which is devnull.
        # But script is internal.
        sys.exit(1)

    snapshot_file = sys.argv[1]
    output_file = sys.argv[2]

    # We rely on profile.py to redirect stderr to output_file, so any crash/traceback is logged automatically.

    if not os.path.exists(snapshot_file):
        # We can write to stderr to log this
        sys.stderr.write(
            f"\n[Memory Analysis Error: Snapshot {snapshot_file} not found]\n"
        )
        sys.exit(1)

    # Load snapshot
    snapshot = tracemalloc.Snapshot.load(snapshot_file)
    top_stats = snapshot.statistics("lineno")

    output = []
    output.append("\n")
    output.append("TOP 50 BY MEMORY SIZE")
    output.append("=" * 80)
    for stat in top_stats[:50]:
        output.append(str(stat))

    output.append("\n")
    output.append("TOP 50 BY MEMORY COUNT")
    output.append("=" * 80)

    # Sort by count
    top_stats_count = sorted(top_stats, key=lambda x: x.count, reverse=True)
    for stat in top_stats_count[:50]:
        output.append(str(stat))

    # Write output
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
