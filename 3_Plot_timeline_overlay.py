import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
csv_file = r"C:\Users\immae\Desktop\Myo\Day5_H6.csv"
df = pd.read_csv(csv_file)

# Sampling frequency
fs = 42.47  # Hz

# No offset: timeline e' relativa all'inizio file
offset_seconds = 0
offset_samples = int(offset_seconds * fs)
df = df.iloc[offset_samples:].reset_index(drop=True)

# Timeline (secondi dall'inizio file)
# Format: (exercise, set, start_sec, end_sec_or_none, label)
timeline = [
    (1, 1, 6, 14, "ese1/set1"),
    (1, 2, 26, 35, "ese1/set2"),
    (2, 1, 47, 56, "ese2/set1"),
    (2, 2, 69, 78, "ese2/set2"),
    (3, 1, 86, 94, "ese3/set1"),
    (3, 2, 104, 111, "ese3/set2"),
    (4, 1, 122, 131, "ese4/set1"),
    (4, 2, 140, 151, "ese4/set2"),
    (5, 1, 158, 166, "ese5/set1"),
    (5, 2, 176, 185, "ese5/set2"),
    (6, 1, 191, 205, "ese6/set1"),
    (6, 2, 213, 230, "ese6/set2"),
    (7, 1, 236, 251, "ese7/set1"),
    (7, 2, 257, 270, "ese7/set2"),
    (8, 1, 283, 295, "ese8/set1"),
    (8, 2, 300, 312, "ese8/set2"),
]


# --- Overlay plot ---
channels = [f"Channel_{i}" for i in range(1, 9)]
colors = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2",
]

time = np.arange(len(df)) / fs
duration_sec = len(df) / fs

fig, axes = plt.subplots(8, 1, figsize=(24, 12), sharex=True)
fig.suptitle("Day5_H6 - Timeline Overlay", fontsize=16, fontweight="bold")

for idx, channel in enumerate(channels):
    ax = axes[idx]
    ax.plot(time, df[channel], "b-", linewidth=0.5, alpha=0.7)

    for ex_num, set_num, start, end, _label in timeline:
        color = colors[ex_num - 1]
        end_sec = duration_sec if end is None else min(end, duration_sec)
        if start >= duration_sec:
            continue

        ax.axvline(start, color=color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(end_sec, color=color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvspan(start, end_sec, alpha=0.16, color=color)

        if idx == 0:
            mid = (start + end_sec) / 2
            ax.text(
                mid,
                ax.get_ylim()[1] * 0.95,
                f"E{ex_num}/S{set_num}",
                ha="center",
                va="top",
                fontsize=7,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor="none"),
            )

    ax.set_ylabel(f"Ch{idx + 1}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration_sec)

axes[-1].set_xlabel("Time (s) - Relative to file start", fontsize=10, fontweight="bold")
plt.tight_layout()

output_png = os.path.join(os.path.dirname(__file__), "Day5_H6_timeline_overlay.png")
plt.savefig(output_png, dpi=150, bbox_inches="tight")
print(f"✓ Salvato: {output_png}")
plt.close()
