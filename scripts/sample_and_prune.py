import random
import sys

filename = 'models/style/grammaticality_confusion.csv'

try:
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    sys.exit(1)

if len(lines) <= 1:
    print("No data lines found.")
    sys.exit(0)

header = lines[0]
data = lines[1:]

sample_size = min(50, len(data))
sampled_indices = set(random.sample(range(len(data)), sample_size))

sampled_lines = []
remaining_lines = []

for i, line in enumerate(data):
    if i in sampled_indices:
        sampled_lines.append(line)
    else:
        remaining_lines.append(line)

# Write back remaining
with open(filename, 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(remaining_lines)

# Write sampled lines to file for the agent to process
with open('sampled_batch.txt', 'w', encoding='utf-8') as f:
    for line in sampled_lines:
        f.write(line)

print(f"Sampled {len(sampled_lines)} lines to sampled_batch.txt")
