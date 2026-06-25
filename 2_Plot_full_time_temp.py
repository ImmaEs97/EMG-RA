"""
Plot the complete EMG recording for all 8 Myo channels.

The script reads one CSV file, assumes a fixed sampling frequency, builds a
time axis in seconds, and saves a full-length plot with one subplot per EMG
channel. This plot is useful before segmentation because it lets you inspect
where the exercises start and end.

Before running:
- place the recording at data/raw/participant_data.csv, or edit input_csv;
- change output_dir if you want to save the PNG somewhere other than results;
- update fs if the recording was collected with a different sampling rate.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration: input file and acquisition frequency
# ---------------------------------------------------------------------------

# Generic project-relative input path.
script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "data", "raw", "participant_data.csv")
df = pd.read_csv(input_csv)

# Fixed sampling frequency.
fs = 50


# ---------------------------------------------------------------------------
# Duration report: verify how long the recording is
# ---------------------------------------------------------------------------

total_samples_full = len(df)
total_duration_full = total_samples_full / fs
print("FULL ACQUISITION:")
print(f"  Total samples: {total_samples_full}")
print(
    f"  Total duration: {total_duration_full:.1f}s = "
    f"{int(total_duration_full // 60)} min {int(total_duration_full % 60)} sec"
)
print()

# Use the full recording.
total_samples = total_samples_full
total_duration = total_duration_full

print("FULL PLOT:")
print(f"Total samples: {total_samples}")
print(
    f"Total duration: {total_duration:.1f}s = "
    f"{int(total_duration // 60)} min {int(total_duration % 60)} sec"
)


# ---------------------------------------------------------------------------
# Plotting: one subplot per EMG channel
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(8, 1, figsize=(20, 12))
fig.suptitle(
    f"Participant - Full recording (0-{total_duration:.1f}s)",
    fontsize=16,
    fontweight="bold",
)

channels = [f"Channel_{i}" for i in range(1, 9)]
time = np.arange(len(df)) / fs

for idx, channel in enumerate(channels):
    axes[idx].plot(time, df[channel], "b-", linewidth=0.5)
    axes[idx].set_ylabel(f"Ch{idx + 1}", fontsize=10, fontweight="bold")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, total_duration)

    if idx == 7:
        axes[idx].set_xlabel("Time (s)", fontsize=10, fontweight="bold")
    else:
        axes[idx].set_xticklabels([])


# ---------------------------------------------------------------------------
# Output: save the figure as PNG
# ---------------------------------------------------------------------------

plt.tight_layout()
output_dir = os.path.join(script_dir, "results")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "participant_full_recording.png")
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\n[OK] Saved: {output_file}")
plt.close()
