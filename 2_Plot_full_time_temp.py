import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Replace this with the input CSV path when running locally.
input_csv = r"PATH_TO_YOUR_INPUT_CSV.csv"
df = pd.read_csv(input_csv)

# Fixed sampling frequency
fs = 50

# Calculate total duration before any limiting
total_samples_full = len(df)
total_duration_full = total_samples_full / fs
print("FULL ACQUISITION:")
print(f"  Total samples: {total_samples_full}")
print(f"  Total duration: {total_duration_full:.1f}s = {int(total_duration_full//60)} min {int(total_duration_full%60)} sec")
print()

# Use full data
total_samples = total_samples_full
total_duration = total_duration_full

print("FULL PLOT:")
print(f"Total samples: {total_samples}")
print(f"Total duration: {total_duration:.1f}s = {int(total_duration//60)} min {int(total_duration%60)} sec")

# Plot all 8 channels for the full duration
fig, axes = plt.subplots(8, 1, figsize=(20, 12))
fig.suptitle(f'Participant - Full recording (0-{total_duration:.1f}s)', fontsize=16, fontweight='bold')


channels = [f'Channel_{i}' for i in range(1, 9)]
time = np.arange(len(df)) / fs  # Time starts from 0

for idx, channel in enumerate(channels):
    axes[idx].plot(time, df[channel], 'b-', linewidth=0.5)
    axes[idx].set_ylabel(f'Ch{idx + 1}', fontsize=10, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, total_duration)
    
    if idx == 7:
        axes[idx].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    else:
        axes[idx].set_xticklabels([])

plt.tight_layout()
# Replace this with the desired output folder when running locally.
output_dir = r"PATH_TO_YOUR_OUTPUT_FOLDER"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'participant_full.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")
plt.close()
