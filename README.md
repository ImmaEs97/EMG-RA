# sEMG-Based Activity Recognition for Rheumatoid Arthritis Monitoring

This repository contains the code associated with the paper:

**Toward Remote Monitoring of Rheumatoid Arthritis Patients via sEMG-Based Activity Recognition**

The study investigates a proof-of-concept framework for using wearable surface electromyography (sEMG) and machine learning to support future remote monitoring of rheumatoid arthritis (RA). sEMG signals were acquired from RA patients during therapeutic hand and wrist exercises inspired by the SARAH exercise framework. The code supports data acquisition, visualization, timeline-based segmentation, feature extraction, exercise classification, and feature-importance analysis.

## Repository Overview

The repository includes five main Python scripts:

| Script | Purpose |
|---|---|
| `1_Data_collection.py` | Acquires 8-channel sEMG data from the Myo Armband in preprocessed mode and saves the recording as a CSV file. |
| `2_Plot_full_time_temp.py` | Plots the full sEMG recording across all eight channels to inspect signal duration and activity periods. |
| `3_Plot_timeline_overlay.py` | Overlays exercise/set timeline annotations on the full recording to visually verify segmentation boundaries. |
| `4_Segment_EMG_data_by_timeline.py` | Segments continuous recordings into exercise-specific CSV files using a manually defined timeline and saves segmentation metadata. |
| `5_Offline_exercise_classifier.py` | Extracts sEMG features and evaluates machine learning models using Leave-One-Subject-Out Cross-Validation. |

## Study Context

The associated paper presents a wearable sEMG-based framework for recognizing therapeutic hand and wrist exercises in patients with rheumatoid arthritis. The experimental protocol includes exercises involving finger abduction/adduction, PIP flexion, MCP flexion, wrist deviation, wrist flexion/extension, and a functional key-turning task.

The classification pipeline uses segmented sEMG recordings, window-based feature extraction, feature standardization within each cross-validation fold, and Leave-One-Subject-Out Cross-Validation. The tested models include Support Vector Machine, Random Forest, Logistic Regression, and XGBoost when available. Majority voting can be used to aggregate window-level predictions into exercise-level predictions.

The code also includes permutation-based feature relevance analysis to identify descriptors that contribute to exercise classification.

## Workflow

A typical workflow is:

1. **Acquire data**

   Use `1_Data_collection.py` to collect sEMG data from the Myo Armband.

2. **Inspect the full recording**

   Use `2_Plot_full_time_temp.py` to visualize all eight sEMG channels and identify the approximate timing of exercise execution.

3. **Verify the exercise timeline**

   Edit the timeline in `3_Plot_timeline_overlay.py` and generate an overlay plot to verify the start and end times of each exercise set.

4. **Segment the recording**

   Edit the participant-specific timeline in `4_Segment_EMG_data_by_timeline.py` and run the script to export one CSV file per exercise, together with timeline and processing metadata.

5. **Run offline classification**

   Use `5_Offline_exercise_classifier.py` to load segmented data, extract features, and evaluate classification models with Leave-One-Subject-Out Cross-Validation.

## Expected Data Format

The scripts expect sEMG recordings in CSV format with eight columns named:

```text
Channel_1, Channel_2, Channel_3, Channel_4, Channel_5, Channel_6, Channel_7, Channel_8
```

Each row corresponds to one time sample. The default sampling frequency used in the scripts is 50 Hz for Myo preprocessed mode, although this can be adapted when needed.

## Suggested Folder Structure

```text
.
├── 1_Data_collection.py
├── 2_Plot_full_time_temp.py
├── 3_Plot_timeline_overlay.py
├── 4_Segment_EMG_data_by_timeline.py
├── 5_Offline_exercise_classifier.py
├── data/
│   ├── raw/
│   ├── Segment/
├── results/
├── README.md
└── LICENSE
```

The exact paths inside the scripts may need to be adapted before running them locally.

## Main Dependencies

The code was developed in Python and uses the following main packages:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
seaborn
tqdm
xgboost
pyomyo
```

`xgboost` is optional and is skipped automatically if it is not installed. `pyomyo` is required only for Myo Armband data acquisition.

## Notes on Data Availability

The repository is intended to share the analysis code. Raw sEMG recordings and clinical data are not included because they may contain participant-related research data and are subject to ethical, privacy, and institutional restrictions.

Users who wish to reproduce the analysis should organize their own data according to the expected CSV structure and adapt the local paths and participant-specific timelines in the scripts.

## Reproducibility Notes

- Timeline values in the segmentation and overlay scripts are participant-specific and should be edited for each recording.
- Feature standardization in the classifier is performed using training subjects only within each Leave-One-Subject-Out fold to reduce data leakage.
- Majority voting can be enabled to aggregate window-level predictions into exercise-level predictions.
- Permutation importance can be used as a post-hoc feature relevance analysis.

## Citation

If you use this code, please cite the associated AVSS 2026 paper:

```bibtex
@inproceedings{esposito2026remoteRAsemg,
  title={Toward Remote Monitoring of Rheumatoid Arthritis Patients via sEMG-Based Activity Recognition},
  author={Esposito, Immacolata and Eken, Defne and Galloway, James and De Benedetto, Egidio and Gionfrida, Letizia},
  booktitle={IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS)},
  year={2026}
}
```

## License

This code is released under the MIT License. See the `LICENSE` file for details.
