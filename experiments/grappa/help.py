import os
import re
from collections import defaultdict

# Directory containing log files
LOG_DIR = "log"

# Regex patterns
file_pattern = re.compile(
    r"(metis|random)_(ogbar|reddit|ogbpr)_(gcn|sage|gat)_(\d)_\d+_R\d+_0\.txt$"
)
infer_line_pattern = re.compile(r"rank0 infer_.*test_acc\|([\d.]+)")

# Data structure: {model_layer: {dataset: test_acc}}
data = defaultdict(dict)

for root, _, files in os.walk(LOG_DIR):
    for fname in files:
        match = file_pattern.search(fname)
        if match:
            partition, dataset, model, layer = match.groups()
            key = f"{model}_{layer}"
            filepath = os.path.join(root, fname)

            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()

                # Find last infer_ line
                infer_lines = [l for l in lines if "infer_" in l and "test_acc" in l]
                if not infer_lines:
                    continue
                last_line = infer_lines[-1]

                # Extract test_acc
                acc_match = infer_line_pattern.search(last_line)
                if acc_match:
                    test_acc = float(acc_match.group(1))
                    data[key][dataset] = test_acc
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# Get all datasets
datasets = sorted({ds for model_data in data.values() for ds in model_data.keys()})

# Print transposed table
header = ["Model_Layer"] + datasets
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")

for model_layer in sorted(data.keys(), key=lambda x: (x.split("_")[0], int(x.split("_")[1]))):
    row = [model_layer]
    for ds in datasets:
        val = data[model_layer].get(ds, "")
        row.append(str(val))
    print("| " + " | ".join(row) + " |")

