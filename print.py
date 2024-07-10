from glob import glob
import numpy as np
import pickle
import sys
from collections import defaultdict

result_root = sys.argv[1]

files = glob(f"{result_root}/*.pkl")
w_jpe = []
wa_jpe = []
w_jpe_per_sequence = defaultdict(list)
wa_jpe_per_sequence = defaultdict(list)
w_jpe_per_interval = defaultdict(list)
wa_jpe_per_interval = defaultdict(list)
for file in files:
    sequence_id = file.split("_")[-2]
    frame_interval = file.split("_")[-1].split(".")[0]
    with open(file, "rb") as f:
        aaa = pickle.load(f)
    w_jpe.append(aaa["eval_metrics"]['w_jpe'])
    wa_jpe.append(aaa["eval_metrics"]['wa_jpe'])
    w_jpe_per_interval[f"{sequence_id}_{frame_interval}"].append(
        aaa["eval_metrics"]['w_jpe'])
    wa_jpe_per_interval[f"{sequence_id}_{frame_interval}"].append(
        aaa["eval_metrics"]['wa_jpe'])
    w_jpe_per_sequence[f"{sequence_id}"].append(aaa["eval_metrics"]['w_jpe'])
    wa_jpe_per_sequence[f"{sequence_id}"].append(aaa["eval_metrics"]['wa_jpe'])

w_jpe = np.concatenate(w_jpe)
wa_jpe = np.concatenate(wa_jpe)
print("overall_metrics:")
print(f"w_jpe: {w_jpe.mean()}")
print(f"wa_jpe: {wa_jpe.mean()}")
print("per_sequence_metrics:")
for seq_name in sorted(w_jpe_per_sequence.keys()):
    print(seq_name)
    w_jpe_per_sequence[seq_name] = np.concatenate(
        w_jpe_per_sequence[seq_name]).mean()
    wa_jpe_per_sequence[seq_name] = np.concatenate(
        wa_jpe_per_sequence[seq_name]).mean()

    print(f"w_jpe: {w_jpe_per_sequence[seq_name]}")
    print(f"wa_jpe: {wa_jpe_per_sequence[seq_name]}")

print("intervals with highest errors:")
for interval_name in sorted(w_jpe_per_interval.keys()):
    w_jpe_per_interval[interval_name] = np.concatenate(
        w_jpe_per_interval[interval_name]).mean()
    wa_jpe_per_interval[interval_name] = np.concatenate(
        wa_jpe_per_interval[interval_name]).mean()
print("w_jpe")
top_5_keys = sorted(w_jpe_per_interval, key=w_jpe_per_interval.get, reverse=True)[:5]

# Print the keys
for key in top_5_keys:
    print(f"{key}: {w_jpe_per_interval[key]}")

print("wa_jpe")
top_5_keys = sorted(wa_jpe_per_interval, key=wa_jpe_per_interval.get, reverse=True)[:5]
# Print the keys
for key in top_5_keys:
    print(f"{key}: {wa_jpe_per_interval[key]}")
