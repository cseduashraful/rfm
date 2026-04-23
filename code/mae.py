import sys
import re
import numpy as np

def compute_mae(file_path):
    pattern = re.compile(
        r"prediction=([0-9eE\.\-]+)\s*\|\s*ground_truth=([0-9eE\.\-]+)"
    )

    preds, gts = [], []

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                preds.append(float(match.group(1)))
                gts.append(float(match.group(2)))

    if not preds:
        print("No valid data found.")
        return

    preds = np.array(preds)
    gts = np.array(gts)

    mae = np.mean(np.abs(preds - gts))

    print(f"MAE: {mae:.6f}")
    print(f"Total samples: {len(preds)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_mae.py <log_file>")
        sys.exit(1)

    compute_mae(sys.argv[1])