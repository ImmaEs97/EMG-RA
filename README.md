# sEMG-Based Activity Recognition for Rheumatoid Arthritis Monitoring

This repository contains the code associated with the paper:

**Toward Remote Monitoring of Rheumatoid Arthritis Patients via sEMG-Based Activity Recognition**

The paper is associated with **AVSS 2026** and investigates a proof-of-concept framework for using wearable surface electromyography (sEMG) and machine learning to support future remote monitoring of rheumatoid arthritis (RA). The code covers the main processing steps used in the study: data acquisition from a Myo armband, signal inspection, manual timeline verification, segmentation, exercise classification, and exploratory association analysis between EMG-derived features and clinical joint assessment data.

## Repository contents

```text
.
├── 1_Data_collection.py
├── 2_Plot_full_time_temp.py
├── 3_Plot_timeline_overlay.py
├── 4_Segment_EMG_data_by_timeline.py
├── 5_Offline_exercise_classifier.py
├── analyze_emg_clinical_groups.py
├── README.md
└── LICENSE
```

## Workflow

### 1. Data collection

`1_Data_collection.py` collects 8-channel EMG samples from a Myo armband and saves them to CSV. The script uses the Myo in preprocessed EMG mode, provides LED feedback during acquisition, periodically saves partial data, and writes a final CSV when data collection is stopped.

Before running, update the output folder:

```python
base_dir = r"PATH_TO_YOUR_OUTPUT_FOLDER"
```

### 2. Full recording inspection

`2_Plot_full_time_temp.py` plots the complete EMG recording for all 8 Myo channels. This step is useful for checking the full acquisition duration and visually identifying the approximate exercise periods before segmentation.

Before running, update:

```python
input_csv = r"PATH_TO_YOUR_INPUT_CSV.csv"
output_dir = r"PATH_TO_YOUR_OUTPUT_FOLDER"
fs = 50
```

### 3. Timeline overlay

`3_Plot_timeline_overlay.py` overlays exercise and set boundaries on the full EMG recording. This is used to visually verify whether the manually defined segmentation timeline matches the signal.

Before running, update:

```python
csv_file = r"PATH_TO_YOUR_INPUT_CSV.csv"
fs = 50
timeline = [
    # (exercise, set, start_sec, end_sec, label)
]
```

### 4. Timeline-based segmentation

`4_Segment_EMG_data_by_timeline.py` segments the full EMG recording into exercise-specific CSV files using a fixed participant timeline. It also saves timeline metadata, summary files, and a configuration file documenting the segmentation used for the run.

Example usage:

```bash
python 4_Segment_EMG_data_by_timeline.py --input path/to/recording.csv --fs 50 --offset 0 --config-label H01
```

The participant-specific timeline should be edited at the top of the script before running.

### 5. Offline exercise classification

`5_Offline_exercise_classifier.py` loads segmented exercise files, applies dynamic trimming and optional per-file normalization, extracts time-domain, frequency-domain, and inter-channel EMG features, and evaluates machine learning classifiers using Leave-One-Subject-Out Cross-Validation.

The script can run Random Forest, SVM, Logistic Regression, and XGBoost when available. It saves summary metrics, confusion matrices, per-class accuracies, prediction files, and permutation-importance tables.

By default, the script expects segmented data in:

```text
data/Segment_v2
```

and saves outputs to:

```text
results
```

Update these paths in the `main()` function if your folder structure is different.

### 6. EMG-clinical association analysis

`analyze_emg_clinical_groups.py` analyzes associations between EMG features and clinical joint assessment data. It computes Spearman correlations between EMG features and clinical joint involvement rates for selected joint groups and applies Benjamini-Hochberg FDR correction. Optionally, it can also perform a two-group Mann-Whitney comparison when case patients are provided.

Example usage:

```bash
python analyze_emg_clinical_groups.py \
  --emg-csv path/to/emg_features.csv \
  --clinical-csv path/to/clinical_joint_data.csv \
  --out-dir results_clinical \
  --joint-groups WRIST_RIGHT,PIP_RIGHT,MCP_RIGHT
```

Optional group comparison:

```bash
python analyze_emg_clinical_groups.py \
  --emg-csv path/to/emg_features.csv \
  --clinical-csv path/to/clinical_joint_data.csv \
  --out-dir results_clinical \
  --case-patients 1,4,7 \
  --case-label HighModerate \
  --control-label LowRemission
```

## Expected input formats

### Raw EMG CSV

The acquisition and preprocessing scripts expect 8 EMG channel columns named:

```text
Channel_1, Channel_2, Channel_3, Channel_4, Channel_5, Channel_6, Channel_7, Channel_8
```

### EMG feature CSV for clinical analysis

The clinical association script expects an EMG feature table with at least:

```text
patient_id, exercise_id
```

plus one or more numeric EMG feature columns, such as:

```text
rms, mav, iemg, wl
```

### Clinical joint CSV

The clinical file should contain:

```text
patient_id, joint_group, right_tender_x_count, right_swollen_x_count
```

Example joint groups are:

```text
WRIST_RIGHT, PIP_RIGHT, MCP_RIGHT
```

## Data availability

Raw EMG recordings and clinical assessment data are not included in this repository because they may contain sensitive participant-related information. Users should run the scripts on their own data or on data made available under the appropriate ethical and data-sharing approvals.

## Dependencies

The scripts require Python 3 and the following packages:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
statsmodels
tqdm
pyomyo
xgboost   # optional, only required for XGBoost classification
```

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels tqdm
```

For Myo data acquisition, install and configure `pyomyo` according to your local Myo setup.

## Notes

Some scripts contain participant-specific timeline examples and should be adapted before use. Before making the repository public, replace local paths, participant labels, and example filenames with generic placeholders where appropriate.

## Citation

If you use this code, please cite the associated paper accepted for AVSS 2026:

```bibtex
@inproceedings{esposito2026remoteRAsemg,
  title={Toward Remote Monitoring of Rheumatoid Arthritis Patients via sEMG-Based Activity Recognition},
  author={Esposito, Immacolata and Eken, Defne and Galloway, James and De Benedetto, Egidio and Gionfrida, Letizia},
  booktitle={IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS)},
  year={2026}
}
```

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
