"""
Offline EMG exercise classifier.
Loads segmented exercise data, extracts time, frequency, and inter-channel
features, and evaluates models with LOSO cross-validation.

The workflow is:
- read one folder of segmented exercise CSV files per participant;
- trim transition samples and optionally normalize each file;
- split each exercise into sliding windows;
- extract selected EMG features from every window;
- evaluate classifiers with Leave-One-Subject-Out cross-validation;
- save confusion matrices, per-class accuracy, summaries, and permutation
  importance tables.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

from scipy import stats


# ---------------------------------------------------------------------------
# Feature selection: choose which EMG descriptors are included in the model
# ---------------------------------------------------------------------------

# Model -> selected feature types
FEATURE_TYPES = [
    'MAD', 'SSC', 'Corr', 'WAMP', 'Peak-Peak', 'Power', 'WL', 'MFL', 'MeanPeakHeight', 
    'HjorthComp', 'Skew', 'HjorthMob', 'RMSratio', 'StdPeakDist', 'RMS', 'SSI', 'MeanPeakDist', 
    'PeakCount', 'BandP_60_80', 'DASDV', 'MDF','BandP_40_60', 'BandP_20_40','BandP_05_20','LogDet',
]

# No duplicate entries
feature_types_to_use = FEATURE_TYPES


# ---------------------------------------------------------------------------
# Feature extraction: convert one EMG window into numeric descriptors
# ---------------------------------------------------------------------------

class EMGFeatureExtractor:
    """Extract features from EMG signals."""
    
    def __init__(self, sampling_rate=50, include_interchannel=True, feature_types_to_use=None):
        """
        Args:
            sampling_rate: Sampling rate in Hz (default 50 Hz in Myo preprocessing mode)
            include_interchannel: Whether to include inter-channel features
            feature_types_to_use: List of feature types to include (e.g. ["MDF", "BandP_40_60", ...])
        """
        self.sampling_rate = sampling_rate
        self.include_interchannel = include_interchannel
        
        # Use only the selected feature types
        self.feature_types_to_use = feature_types_to_use
    
    def extract_time_features(self, channel_data):
        """Time-domain features, including peak-based descriptors."""
        features = []
        N = len(channel_data)
        diff_signal = np.diff(channel_data)
        from scipy.stats import skew, kurtosis
        from scipy.signal import find_peaks
        abs_signal = np.abs(channel_data)
        min_distance = max(1, int(0.05 * self.sampling_rate))
        prominence = 0.2 * np.max(abs_signal)
        peaks, props = find_peaks(abs_signal, distance=min_distance, prominence=prominence)
        peak_count = len(peaks)
        if peak_count >= 2:
            peak_diffs = np.diff(peaks) / self.sampling_rate  # distance in seconds
            mean_peak_dist = float(np.mean(peak_diffs))
            std_peak_dist = float(np.std(peak_diffs))
            side_peak_dist = np.max(peak_diffs) if len(peak_diffs) > 0 else 0.0
        else:
            mean_peak_dist = 0.0
            std_peak_dist = 0.0
            side_peak_dist = 0.0
        if peak_count > 0:
            mean_peak_height = float(np.mean(props.get('prominences', abs_signal[peaks])))
        else:
            mean_peak_height = 0.0

        # Piecewise MMAV1/MMAV2 weights
        w1 = np.ones(N)
        w2 = np.ones(N)
        w1[:int(0.25*N)] = 0.5
        w1[int(0.75*N):] = 0.5
        mmav1 = np.sum(np.abs(channel_data) * w1) / N
        w2[:int(0.25*N)] = 1.5
        w2[int(0.75*N):] = 1.5
        mmav2 = np.sum(np.abs(channel_data) * w2) / N

        feature_map = {
            'RMS': np.sqrt(np.mean(channel_data**2)),
            'MAV': np.mean(np.abs(channel_data)),
            'Var': np.var(channel_data),
            'WL': np.sum(np.abs(np.diff(channel_data))),
            'SSC': np.sum(np.diff(np.sign(diff_signal)) != 0),
            'Peak-Peak': np.max(channel_data) - np.min(channel_data),
            'MaxAbs': np.max(np.abs(channel_data)),
            'Skew': skew(channel_data),
            'Kurt': kurtosis(channel_data),
            'EnvSlope': np.polyfit(np.arange(len(np.abs(channel_data))), np.abs(channel_data), 1)[0],
            'IEMG': np.sum(np.abs(channel_data)),
            'LogDet': np.exp(np.mean(np.log(np.abs(channel_data) + 1e-12))),
            'SSI': np.sum(channel_data ** 2),
            'WAMP': np.sum(np.abs(diff_signal) > 0.1 * (np.std(channel_data) + 1e-12)),
            'HjorthMob': np.sqrt(np.var(diff_signal) / (np.var(channel_data) + 1e-12)),
            'HjorthComp': (np.sqrt(np.var(np.diff(diff_signal)) / (np.var(diff_signal) + 1e-12)) / (np.sqrt(np.var(diff_signal) / (np.var(channel_data) + 1e-12)) + 1e-12)),
            'EMAV': np.sum(np.abs(channel_data) * np.arange(1, N+1)) / (N * (N+1)/2),
            'DASDV': np.sqrt(np.mean(np.diff(channel_data)**2)),
            'EWL': np.sum(np.abs(np.diff(channel_data)) * np.arange(1, N)) / (N * (N-1)/2) if N > 1 else 0.0,
            'MMAV1': mmav1,
            'MMAV2': mmav2,
            'MFL': np.sum(np.abs(np.diff(channel_data))) / N,
            'MYOP': np.sum(np.abs(channel_data) > 0.25 * np.max(np.abs(channel_data))) / N,
            # Peak features
            'PeakCount': peak_count,
            'MeanPeakDist': mean_peak_dist,
            'StdPeakDist': std_peak_dist,
            'MeanPeakHeight': mean_peak_height,
            'SidePeakDist': side_peak_dist,
        }
        for k in feature_map:
            if (self.feature_types_to_use is None) or (k in self.feature_types_to_use):
                features.append(feature_map[k])
        return features
    
    def extract_frequency_features(self, channel_data):
        """Frequency-domain features."""
        features = []
        fft_vals = np.fft.rfft(channel_data)
        fft_freq = np.fft.rfftfreq(len(channel_data), 1.0/self.sampling_rate)
        power_spectrum = np.abs(fft_vals)**2
        mnf = np.sum(fft_freq * power_spectrum) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0
        cumsum = np.cumsum(power_spectrum)
        mdf_idx = np.where(cumsum >= cumsum[-1] / 2)[0]
        mdf = fft_freq[mdf_idx[0]] if len(mdf_idx) > 0 else 0
        total_power = np.sum(power_spectrum)
        psd_norm = power_spectrum / (total_power + 1e-12)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        spec_spread = np.sqrt(np.sum(((fft_freq - mnf) ** 2) * psd_norm))
        nyq = self.sampling_rate / 2.0
        band_fracs = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
        band_powers = []
        for lo_f, hi_f in band_fracs:
            lo = lo_f * nyq
            hi = hi_f * nyq
            mask = (fft_freq >= lo) & (fft_freq < hi)
            bp = np.sum(power_spectrum[mask]) / (total_power + 1e-12)
            band_powers.append(bp)
        # Feature map
        feature_map = {
            'MNF': mnf,
            'MDF': mdf,
            'Power': total_power,
            'SpecEntropy': spec_entropy,
            'SpecSpread': spec_spread,
            'BandP_05_20': band_powers[0] if len(band_powers) > 0 else 0.0,
            'BandP_20_40': band_powers[1] if len(band_powers) > 1 else 0.0,
            'BandP_40_60': band_powers[2] if len(band_powers) > 2 else 0.0,
            'BandP_60_80': band_powers[3] if len(band_powers) > 3 else 0.0,
        }
        for k in feature_map:
            if (self.feature_types_to_use is None) or (k in self.feature_types_to_use):
                features.append(feature_map[k])
        return features
    
    def extract_features(self, signal):
        """
        Extract only the selected features from all channels.
        """
        features = []
        for ch in range(signal.shape[1]):
            channel_data = signal[:, ch]
            # Time and frequency: selected features only
            features.extend(self.extract_time_features(channel_data))
            features.extend(self.extract_frequency_features(channel_data))
        if self.include_interchannel:
            features.extend(self.extract_interchannel_features(signal))
        return np.array(features)

    def extract_interchannel_features(self, signal):
        """Inter-channel features: selected features only."""
        inter_features = []
        rms_per_ch = [np.sqrt(np.mean(signal[:, ch] ** 2)) for ch in range(signal.shape[1])]
        for i, j in combinations(range(signal.shape[1]), 2):
            xi = signal[:, i]
            xj = signal[:, j]
            denom = (np.std(xi) * np.std(xj)) + 1e-12
            # Selected inter-channel features only
            if (self.feature_types_to_use is None) or ("Corr" in self.feature_types_to_use):
                corr = float(np.corrcoef(xi, xj)[0, 1]) if denom > 0 else 0.0
                inter_features.append(corr)
            if (self.feature_types_to_use is None) or ("RMSratio" in self.feature_types_to_use):
                rms_ratio = rms_per_ch[i] / (rms_per_ch[j] + 1e-12)
                inter_features.append(rms_ratio)
            if (self.feature_types_to_use is None) or ("MAD" in self.feature_types_to_use):
                mad = float(np.mean(np.abs(xi - xj)))
                inter_features.append(mad)
        return inter_features

    def get_feature_names(self, num_channels: int) -> list[str]:
        """Return ordered feature names matching extract_features output."""
        time_features = [
            'RMS', 'MAV', 'Var', 'WL', 'SSC', 'Peak-Peak', 'MaxAbs', 'Skew', 'Kurt',
            'EnvSlope', 'IEMG', 'LogDet', 'SSI', 'WAMP', 'HjorthMob', 'HjorthComp',
            'EMAV', 'DASDV', 'EWL', 'MMAV1', 'MMAV2', 'MFL', 'MYOP',
            'PeakCount', 'MeanPeakDist', 'StdPeakDist', 'MeanPeakHeight', 'SidePeakDist',
        ]
        freq_features = [
            'MNF', 'MDF', 'Power', 'SpecEntropy', 'SpecSpread',
            'BandP_05_20', 'BandP_20_40', 'BandP_40_60', 'BandP_60_80',
        ]
        inter_features = ['Corr', 'RMSratio', 'MAD']

        def is_selected(name: str) -> bool:
            return (self.feature_types_to_use is None) or (name in self.feature_types_to_use)

        names = []
        for ch in range(num_channels):
            ch_prefix = f"ch{ch+1}"
            for feat in time_features:
                if is_selected(feat):
                    names.append(f"{ch_prefix}_{feat}")
            for feat in freq_features:
                if is_selected(feat):
                    names.append(f"{ch_prefix}_{feat}")

        if self.include_interchannel:
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    pair_prefix = f"ch{ i+1 }_ch{ j+1 }"
                    for feat in inter_features:
                        if is_selected(feat):
                            names.append(f"{pair_prefix}_{feat}")

        return names


# ---------------------------------------------------------------------------
# Classifier pipeline: loading, windowing, training, and evaluation
# ---------------------------------------------------------------------------

class ExerciseClassifier:
    """EMG exercise classifier."""
    
    def __init__(self, data_dir, results_dir=None, window_size=2500, overlap=0.50, sampling_rate=50,
                 group_exercises=True, skip_ex8=True,
                 normalize=True, include_interchannel=True,
                 use_voting=False,
                 include_raw_amplitude=False, raw_amp_mode="global",
                 feature_types_to_use=feature_types_to_use,
                 compute_permutation_importance=True, perm_n_repeats=5, perm_top_n=20):
        """
        Args:
            data_dir: Path to the folder containing segmented data
            results_dir: Path to the folder used to save results
            window_size: Window size in milliseconds (default 2500 ms = 2.5 s)
            overlap: Window overlap fraction (default 0.50 = 50%)
            sampling_rate: Sampling rate in Hz (default 50 Hz)
        """
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be in [0,1). Received: {overlap}")

        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir) if results_dir else Path(data_dir).parent / "results"
        self.results_dir.mkdir(exist_ok=True)  # Create the folder if it does not exist
        # Windowing parameters
        self.sampling_rate = sampling_rate
        self.window_size_ms = window_size
        self.window_size_samples = int((window_size / 1000.0) * sampling_rate)
        self.overlap = overlap
        self.X = None
        self.y = None
        self.patient_ids = None
        self.group_ids = None
        # Feature extractor: selected feature types only
        self.feature_extractor = EMGFeatureExtractor(
            sampling_rate=sampling_rate,
            include_interchannel=include_interchannel,
            feature_types_to_use=feature_types_to_use
        )
        self.group_exercises = group_exercises
        self.skip_ex8 = skip_ex8  # Skip exercise 8 if requested
        self.normalize = normalize
        self.use_voting = use_voting
        self.include_raw_amplitude = include_raw_amplitude
        self.raw_amp_mode = raw_amp_mode
        self.compute_permutation_importance = compute_permutation_importance
        self.perm_n_repeats = perm_n_repeats
        self.perm_top_n = perm_top_n
        self.feature_names = None
        # Configuration log
        print("Windowing configuration:")
        print(f"  - Window: {self.window_size_ms} ms ({self.window_size_samples} samples)")
        print(f"  - Overlap: {int(overlap*100)}%")
        print(f"  - Sampling rate: {self.sampling_rate}Hz")
        print("  - Initial/final dynamic trim: ON")
        print(f"  - Amplitude normalization: {'ON' if self.normalize else 'OFF'}")
        print(f"  - Inter-channel features: {'ON' if include_interchannel else 'OFF'}")
        print("  - Sliding-window sampling: ON")
        print(f"  - Exercise voting: {'ON' if self.use_voting else 'OFF'}")
        print(f"  - Raw amplitude feature: {'ON' if self.include_raw_amplitude else 'OFF'}")
        if self.include_raw_amplitude:
            print(f"    - Raw amp mode: {self.raw_amp_mode}")
        print()
    
    def _majority_vote(self, y_true, y_pred, group_ids):
        """Majority vote per exercise group."""
        grouped = {}
        for i, gid in enumerate(group_ids):
            if gid not in grouped:
                grouped[gid] = {'true': y_true[i], 'pred': []}
            grouped[gid]['pred'].append(y_pred[i])

        y_true_v = []
        y_pred_v = []
        for entry in grouped.values():
            preds = np.asarray(entry['pred'])
            values, counts = np.unique(preds, return_counts=True)
            max_count = np.max(counts)
            top_values = values[counts == max_count]
            pred_label = int(np.min(top_values))  # tie-breaker: lowest label
            y_true_v.append(entry['true'])
            y_pred_v.append(pred_label)

        return np.array(y_true_v), np.array(y_pred_v)

    def _get_dynamic_trim_seconds(self, duration_s):
        # --- DYNAMIC TRIM ---
        # Remove start/end transitions based on signal duration.
        # This helps suppress artifacts near exercise boundaries.
        if duration_s <= 12.0:
            return 0.0
        if duration_s <= 20.0:
            return 1.0
        return 2.0

    def _standardize_train_test(self, X_train, X_test):
        # --- STANDARDIZATION ---
        # Compute mean and std only on the training set, then apply to train and test.
        # This avoids data leakage.
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-12
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        print("[DEBUG] Standardization: train mean = {:.4f}, mean train variance = {:.4f}".format(np.mean(X_train_scaled), np.mean(np.var(X_train_scaled, axis=0))))
        return X_train_scaled, X_test_scaled

    def load_data(self):
        """
        --- DATA LOADING ---
        Load all segmented data, apply dynamic trim, per-file normalization,
        sliding-window sampling, and feature extraction.
        - Dynamic trim: removes start/end transitions.
        - Normalization: per-channel z-score on each file only.
        - Windowing: sliding windows only.
        - Features: time, frequency, optional inter-channel, and raw amplitude.
        """
        print("Loading data...")
        
        all_features = []
        all_labels = []
        all_patients = []
        all_group_ids = []
        skipped_files = []
        empty_patients = []
        
        # Cache sampling rate per participant
        patient_sampling_rates = {}
        
        def get_sampling_rate(patient_folder):
            """Read the sampling frequency from processing_info.txt."""
            info_file = next(patient_folder.glob("*_processing_info.txt"), None)
            if info_file is not None and info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        for line in f:
                            if 'Sampling frequency:' in line:
                                fs_str = line.split(':')[1].strip().split()[0]
                                return float(fs_str)
                except:
                    pass
            return self.sampling_rate  # Default

        # --- PARTICIPANT LOADING ---
        patient_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        for patient_folder in tqdm(patient_folders, desc="Loading participants"):
            # Extract the participant ID from the folder name
            patient_id = patient_folder.name.split('_')[0]
            # Read the correct sampling frequency for this participant
            if patient_id not in patient_sampling_rates:
                patient_sampling_rates[patient_id] = get_sampling_rate(patient_folder)
            patient_fs = patient_sampling_rates[patient_id]
            patient_window_samples = int((self.window_size_ms / 1000.0) * patient_fs)
            self.feature_extractor.sampling_rate = patient_fs
            exercise_files = sorted(patient_folder.glob("*_ex*.csv"))
            patient_loaded = 0
            for ex_file in exercise_files:
                # Extract the exercise number from the file name
                try:
                    ex_num = int(ex_file.stem.split('_ex')[1])
                except:
                    skipped_files.append(ex_file.name)
                    continue
                # Skip exercise 8 if requested
                if self.skip_ex8 and ex_num == 8:
                    continue
                # Grouping: 3->2, 4/5->3, otherwise keep ex_num
                if self.group_exercises:
                    if ex_num == 3:
                        grouped_label = 2
                    elif ex_num in (4, 5):
                        grouped_label = 3
                    else:
                        grouped_label = ex_num
                else:
                    grouped_label = ex_num
                try:
                    emg_data = pd.read_csv(ex_file).values
                    group_id = f"{patient_id}_{ex_file.stem}"
                    if emg_data.shape[0] < patient_window_samples or emg_data.shape[1] != 8:
                        skipped_files.append(f"{ex_file.name} (insufficient data: {emg_data.shape})")
                        continue
                    # --- DYNAMIC TRIM ---
                    duration_s = emg_data.shape[0] / patient_fs
                    trim_seconds = self._get_dynamic_trim_seconds(duration_s)
                    trim_samples = int(trim_seconds * patient_fs)
                    if emg_data.shape[0] > 2 * trim_samples + patient_window_samples:
                        emg_data = emg_data[trim_samples:-trim_samples, :]
                    # Save a pre-normalization copy for raw amplitude
                    emg_raw_amp = emg_data.copy()
                    # --- PER-FILE NORMALIZATION ---
                    # Per-file normalization (z-score per channel, no leakage)
                    if self.normalize:
                       mean = np.mean(emg_data, axis=0)
                       std = np.std(emg_data, axis=0) + 1e-12
                       emg_data = (emg_data - mean) / std
                    # --- WINDOWING (sliding windows only) ---
                    step_size_patient = max(1, int(patient_window_samples * (1 - self.overlap)))
                    num_windows = (emg_data.shape[0] - patient_window_samples) // step_size_patient + 1
                    for i in range(num_windows):
                        start_idx = i * step_size_patient
                        end_idx = start_idx + patient_window_samples
                        window = emg_data[start_idx:end_idx, :]
                        # --- SELECTED FEATURES ---
                        features = self.feature_extractor.extract_features(window)
                        if self.include_raw_amplitude:
                            raw_window = emg_raw_amp[start_idx:end_idx, :]
                            if self.raw_amp_mode == "per_channel":
                                raw_amp = np.sqrt(np.mean(raw_window ** 2, axis=0))
                            else:
                                raw_amp = np.array([np.sqrt(np.mean(raw_window ** 2))])
                            features = np.concatenate([features, raw_amp])
                        all_features.append(features)
                        all_labels.append(grouped_label)
                        all_patients.append(int(patient_id))
                        all_group_ids.append(group_id)
                        patient_loaded += 1
                except Exception as e:
                    skipped_files.append(f"{ex_file.name} ({str(e)})")
                    continue
            if patient_loaded == 0:
                empty_patients.append(patient_id)
        self.X = np.array(all_features)
        self.y = np.array(all_labels)
        self.patient_ids = np.array(all_patients)
        self.group_ids = np.array(all_group_ids)
        self.feature_names = self.feature_extractor.get_feature_names(num_channels=8)
        print(f"\n{'='*60}")
        print("DATA LOADING STATISTICS")
        print(f"{'='*60}")
        print(f"  - Total samples: {len(self.X)}")
        print(f"  - Number of features: {self.X.shape[1]}")

        # Compute and print the sample/feature ratio
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        ratio = n_samples / n_features if n_features > 0 else 0
        print(f"  - Sample/feature ratio: {ratio:.2f}")
        if ratio < 5:
            print("[WARNING] The number of features is high relative to the number of samples. Risk of overfitting!")

        print(f"  - Unique participants: {len(np.unique(self.patient_ids))}")
        print(f"  - Unique exercises: {sorted(np.unique(self.y))}")
        print("\nExercises per participant:")
        for pid in sorted(np.unique(self.patient_ids)):
            count = np.sum(self.patient_ids == pid)
            exercises = sorted(self.y[self.patient_ids == pid])
            print(f"  Participant {pid:2d}: {count} exercises -> {exercises}")
        print("\nExercise distribution:")
        for ex in sorted(np.unique(self.y)):
            count = np.sum(self.y == ex)
            print(f"    Exercise {ex}: {count} samples")
        if skipped_files:
            print(f"\n[WARNING] Skipped files ({len(skipped_files)}):")
            for f in skipped_files[:10]:
                print(f"    - {f}")
            if len(skipped_files) > 10:
                print(f"    ... and {len(skipped_files)-10} more files")
        if empty_patients:
            print(f"\n[WARNING] Participants without valid data: {', '.join(empty_patients)}")
        print(f"{'='*60}\n")
        return self.X, self.y, self.patient_ids
    
    def cross_validate_loso(self, shuffle_groups=False, random_state=42, model_to_use=None):
        """
        --- LOSO CROSS-VALIDATION ---
        Leave-One-Subject-Out CV: one participant at a time is used for testing.
        - Each fold uses one held-out participant as the test set.
        - Normalization is fitted on the training split only.
        - Voting aggregates windows per exercise in the test set.
        """
        if self.X is None:
            self.load_data()

        print(f"\n{'='*60}")
        print("LOSO cross-validation (Leave-One-Subject-Out)")
        print(f"{'='*60}\n")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        try:
            from xgboost import XGBClassifier
            xgb_available = True
        except ImportError:
            xgb_available = False

        models = {
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'LDA': LinearDiscriminantAnalysis()
        }
        if xgb_available:
            models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

        if model_to_use is None:
            raise ValueError("model_to_use must be specified (e.g. 'Random Forest')")
        if model_to_use not in models:
            raise ValueError(f"Model '{model_to_use}' is not supported. Choose from: {list(models.keys())}")
        model = models[model_to_use]

        results = {}
        # Test folds across all participants (full LOSO)
        unique_patients = np.unique(self.patient_ids)
        test_patients = unique_patients
        if shuffle_groups:
            rng = np.random.RandomState(random_state)
            test_patients = rng.permutation(test_patients)

        from sklearn.preprocessing import LabelEncoder

        print(f"\n{'='*60}")
        print(f"Model: {model_to_use}")
        print(f"{'='*60}")

        fold_accuracies = []
        all_y_true = []
        all_y_pred = []
        perm_importance_records = []
        perm_top_counts = {}

        for fold_idx, test_patient in enumerate(test_patients, start=1):
            # --- TRAIN/TEST SPLIT ---
            test_mask = self.patient_ids == test_patient
            train_idx = np.where(self.patient_ids != test_patient)[0]
            test_idx = np.where(test_mask)[0]
            print(f"\nFold {fold_idx}/{len(test_patients)} - Test participant {test_patient}")

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # --- NORMALIZATION ---
            X_train_scaled, X_test_scaled = self._standardize_train_test(X_train, X_test)

            # --- TRAINING ---
            if model_to_use == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                y_test_for_eval = y_test
            elif model_to_use == 'XGBoost':
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                y_test_enc = le.transform(y_test)
                model.fit(X_train_scaled, y_train_enc)
                y_pred_enc = model.predict(X_test_scaled)
                y_pred_proba_enc = model.predict_proba(X_test_scaled)
                y_pred = le.inverse_transform(y_pred_enc)
                y_pred_proba = y_pred_proba_enc[:, le.transform(le.classes_)]
                y_test_for_eval = y_test
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    # Fallback: one-hot predictions if the model does not expose probabilities.
                    classes = np.unique(y_train)
                    class_to_idx = {c: i for i, c in enumerate(classes)}
                    y_pred_proba = np.zeros((len(y_pred), len(classes)), dtype=float)
                    for i, pred in enumerate(y_pred):
                        y_pred_proba[i, class_to_idx[pred]] = 1.0
                y_test_for_eval = y_test

            # --- MAJORITY VOTING ---
            if self.use_voting:
                group_ids_test = self.group_ids[test_idx]
                y_true_eval, y_pred_eval = self._majority_vote(
                    y_test_for_eval,
                    y_pred,
                    group_ids_test,
                )
                acc = accuracy_score(y_true_eval, y_pred_eval)
                fold_accuracies.append(acc)
                all_y_true.extend(y_true_eval)
                all_y_pred.extend(y_pred_eval)
                print(f"  Accuracy (majority voting): {acc:.3f} across {len(y_true_eval)} groups")
            else:
                acc = accuracy_score(y_test_for_eval, y_pred)
                fold_accuracies.append(acc)
                all_y_true.extend(y_test_for_eval)
                all_y_pred.extend(y_pred)
                print(f"  Accuracy: {acc:.3f}")

            if self.compute_permutation_importance:
                perm = permutation_importance(
                    model,
                    X_test_scaled,
                    y_test_for_eval,
                    n_repeats=self.perm_n_repeats,
                    random_state=42,
                    scoring="accuracy",
                )
                feature_names = self.feature_names or [f"f{i}" for i in range(X_test_scaled.shape[1])]
                for feat_name, mean_imp, std_imp in zip(
                    feature_names, perm.importances_mean, perm.importances_std
                ):
                    perm_importance_records.append({
                        "fold": fold_idx,
                        "feature": feat_name,
                        "importance_mean": float(mean_imp),
                        "importance_std": float(std_imp),
                    })

                top_idx = np.argsort(perm.importances_mean)[::-1][: self.perm_top_n]
                for i in top_idx:
                    fname = feature_names[i]
                    perm_top_counts[fname] = perm_top_counts.get(fname, 0) + 1

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        results[model_to_use] = {
            'accuracies': fold_accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'y_true': all_y_true,
            'y_pred': all_y_pred
        }

        if self.compute_permutation_importance and perm_importance_records:
            perm_df = pd.DataFrame(perm_importance_records)
            perm_summary = (
                perm_df.groupby("feature", as_index=False)["importance_mean"]
                .mean()
                .rename(columns={"importance_mean": "importance_mean_across_folds"})
            )
            perm_summary["top_count"] = perm_summary["feature"].map(perm_top_counts).fillna(0).astype(int)
            perm_summary = perm_summary.sort_values(
                ["top_count", "importance_mean_across_folds"], ascending=[False, False]
            )

            results[model_to_use]["perm_importance_per_fold"] = perm_df
            results[model_to_use]["perm_importance_summary"] = perm_summary

        print(f"\n{'='*60}")
        print(f"{model_to_use} results (LOSO):")
        print(f"  Mean accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
        print(f"{'='*60}")

        return results

    # -----------------------------------------------------------------------
    # Reporting helpers: text summaries and plots
    # -----------------------------------------------------------------------

    def print_summary(self, results):
        """Print a summary of the results."""
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
        
        print(f"{'Model':<20} {'Accuracy':<15} {'Std Dev':<10}")
        print(f"{'-'*50}")
        for model_name, res in sorted_results:
            print(f"{model_name:<20} {res['mean_accuracy']:.4f}          +/- {res['std_accuracy']:.4f}")
        
        # Single model: no model comparison
        if len(sorted_results) == 1:
            model_name, res = sorted_results[0]
            print(f"\nModel used: {model_name} ({res['mean_accuracy']:.3f})")

        # Save to file as well
        lines = []
        lines.append(f"{'='*60}")
        lines.append("RESULTS SUMMARY")
        lines.append(f"{'='*60}\n")
        lines.append(f"{'Model':<20} {'Accuracy':<15} {'Std Dev':<10}")
        lines.append(f"{'-'*50}")
        for model_name, res in sorted_results:
            lines.append(f"{model_name:<20} {res['mean_accuracy']:.4f}          +/- {res['std_accuracy']:.4f}")
        if len(sorted_results) == 1:
            model_name, res = sorted_results[0]
            lines.append(f"\nModel used: {model_name} ({res['mean_accuracy']:.3f})")
        filename = self.results_dir / 'summary_v5.txt'
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        print(f"Summary saved to: {filename}")
        
    def plot_confusion_matrix(self, results, model_name):
        """Plot the confusion matrix for one model."""
        res = results[model_name]
        
        labels = sorted(set(res['y_true']) | set(res['y_pred']))
        cm = confusion_matrix(res['y_true'], res['y_pred'], labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Exercise')
        plt.xlabel('Predicted Exercise')
        plt.tight_layout()
        filename = self.results_dir / f'confusion_matrix_{model_name.replace(" ", "_")}_v5.png'
        plt.savefig(filename, dpi=150)
        print(f"Confusion matrix saved: {filename}")
        plt.close()

    def print_per_class_accuracy(self, results, model_name):
        """Print per-class accuracy (percentage of correct samples per exercise)."""
        res = results[model_name]
        labels = sorted(set(res['y_true']) | set(res['y_pred']))
        cm = confusion_matrix(res['y_true'], res['y_pred'], labels=labels)

        print(f"\nPer-class accuracy - {model_name}:")
        lines = []
        for idx, label in enumerate(labels):
            total = np.sum(cm[idx, :])
            correct = cm[idx, idx]
            acc = (correct / total) * 100 if total > 0 else 0.0
            line = f"  Exercise {label}: {acc:.4f}% ({correct}/{total})"
            print(line)
            lines.append(line)
        filename = self.results_dir / f'per_class_accuracy_{model_name.replace(" ", "_")}_v5.txt'
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        print(f"Per-class accuracy saved to: {filename}")
        
    def plot_results_comparison(self, results):
        """Plot the model comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Mean accuracy with error bars
        models = list(results.keys())
        means = [results[m]['mean_accuracy'] for m in models]
        stds = [results[m]['std_accuracy'] for m in models]
        
        ax1.bar(range(len(models)), means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison (Mean +/- Std Dev)')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Accuracy per fold
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy per Fold')
        for model_name, res in results.items():
            ax2.plot(range(1, len(res['accuracies'])+1), res['accuracies'], 
                    marker='o', label=model_name)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = self.results_dir / 'model_comparison_v5.png'
        plt.savefig(filename, dpi=150)
        print(f"Comparison plot saved: {filename}")
        plt.close()


# ---------------------------------------------------------------------------
# Script entry point: configure data folders and run selected models
# ---------------------------------------------------------------------------

def main():
    # Path to the folder with segmented data
    data_dir = Path(__file__).parent / "data" / "segmented"
    results_dir = Path(__file__).parent / "results"
    
    # Print the feature set used by the classifier
    print(f"Feature set: {feature_types_to_use}")
    models_to_run = ["Random Forest", "SVM", "Logistic Regression"]
    try:
        import xgboost  # noqa: F401
        models_to_run.append("XGBoost")
    except ImportError:
        print("XGBoost is not installed: skipping XGBoost.")

    results = {}
    for model_name in models_to_run:
        print(f"\n--- Running {model_name} ---")
        classifier = ExerciseClassifier(
            data_dir,
            results_dir,
            window_size=2500,
            overlap=0.50,
            sampling_rate=42,
            group_exercises=True,
            skip_ex8=True,
            normalize=True,
            include_interchannel=True,
            use_voting=True,
            include_raw_amplitude=False,
            raw_amp_mode="per_channel",
            feature_types_to_use=feature_types_to_use
        )
        classifier.load_data()
        res = classifier.cross_validate_loso(shuffle_groups=True, random_state=42, model_to_use=model_name)
        results.update(res)
        # Save the confusion matrix and per-class accuracy with the v5 suffix
        res_labels = sorted(set(results[model_name]['y_true']) | set(results[model_name]['y_pred']))
        cm = confusion_matrix(results[model_name]['y_true'], results[model_name]['y_pred'], labels=res_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=res_labels, yticklabels=res_labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Exercise')
        plt.xlabel('Predicted Exercise')
        plt.tight_layout()
        filename_cm = classifier.results_dir / f'confusion_matrix_{model_name.replace(" ", "_")}_v5.png'
        plt.savefig(filename_cm, dpi=150)
        print(f"Confusion matrix saved: {filename_cm}")
        plt.close()

        # Save y_true and y_pred for external scripts (SVM and XGBoost only)
        if model_name in ["SVM", "XGBoost"]:
            y_true_path = classifier.results_dir / f"y_true_{model_name}.txt"
            y_pred_path = classifier.results_dir / f"y_pred_{model_name}.txt"
            np.savetxt(y_true_path, np.array(results[model_name]['y_true']), fmt='%d')
            np.savetxt(y_pred_path, np.array(results[model_name]['y_pred']), fmt='%d')
            print(f"Saved: {y_true_path}, {y_pred_path}")

        # Per-class accuracy with the v5 suffix
        lines = []
        for idx, label in enumerate(res_labels):
            total = np.sum(cm[idx, :])
            correct = cm[idx, idx]
            acc = (correct / total) * 100 if total > 0 else 0.0
            line = f"  Exercise {label}: {acc:.1f}% ({correct}/{total})"
            print(line)
            lines.append(line)
        filename_acc = classifier.results_dir / f'per_class_accuracy_{model_name.replace(" ", "_")}_v5.txt'
        with open(filename_acc, 'w') as f:
            f.write("\n".join(lines))
        print(f"Per-class accuracy saved to: {filename_acc}")

        if "perm_importance_summary" in results[model_name]:
            perm_summary_path = classifier.results_dir / f"perm_importance_summary_{model_name.replace(' ', '_')}_v5.csv"
            perm_per_fold_path = classifier.results_dir / f"perm_importance_per_fold_{model_name.replace(' ', '_')}_v5.csv"
            results[model_name]["perm_importance_summary"].to_csv(perm_summary_path, index=False)
            results[model_name]["perm_importance_per_fold"].to_csv(perm_per_fold_path, index=False)
            print(f"Permutation importance summary saved: {perm_summary_path}")
            print(f"Permutation importance per fold saved: {perm_per_fold_path}")

    # Save the v5 summary
    summary_lines = []
    summary_lines.append(f"{'='*60}")
    summary_lines.append("V5 RESULTS SUMMARY")
    summary_lines.append(f"{'='*60}\n")
    summary_lines.append(f"{'Model':<20} {'Accuracy':<15} {'Std Dev':<10}")
    summary_lines.append(f"{'-'*50}")
    for model_name in models_to_run:
        res = results.get(model_name, None)
        if res:
            summary_lines.append(f"{model_name:<20} {res['mean_accuracy']:.3f}          +/- {res['std_accuracy']:.3f}")
    summary_path = results_dir / 'summary_v5.txt'
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to {summary_path}")
    print("\n[OK] Analysis completed!")
    print(f"Plots and metrics saved in: {Path(__file__).parent}")


if __name__ == "__main__":
    main()
