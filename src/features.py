#usage: python -m src.features

import os
import csv
import numpy as np


def read_csv(filepath):
    """
    Read a sensor CSV file produced by data_generator.py.
    Skips the '#' metadata header lines, returns:
        timestamps, accel (N,3), gyro (N,3), fs, activity
    """
    timestamps = []
    accel = []
    gyro = []
    fs = None
    activity = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                
                content = line[1:]
                for token in content.split(","):
                    token = token.strip()
                    while token.startswith("#"):
                        token = token[1:].strip()
                    if token.startswith("sampling_rate_hz="):
                        fs = float(token.split("=")[1])
                    elif token.startswith("activity="):
                        activity = token.split("=")[1]
                continue
            if line.startswith("timestamp"):
                continue  # header row
            parts = line.split(",")
            timestamps.append(float(parts[0]))
            accel.append([float(parts[1]), float(parts[2]), float(parts[3])])
            gyro.append([float(parts[4]), float(parts[5]), float(parts[6])])

    return (np.array(timestamps),
            np.array(accel),
            np.array(gyro),
            fs,
            activity)


def _accel_magnitude(accel):
    """Euclidean norm of the 3-axis accelerometer signal."""
    return np.sqrt(np.sum(accel ** 2, axis=1))


def extract_features(accel, gyro, fs, window_sec=2.0):
    """
    Extract features from sliding windows.

    Parameters
    ----------
    accel : (N, 3) array
    gyro  : (N, 3) array
    fs    : float, sampling rate in Hz
    window_sec : float, window length in seconds

    Returns
    -------
    X : (n_windows, n_features) array
    feature_names : list of str
    """
    n_samples = len(accel)
    win_len = int(window_sec * fs)
    if win_len < 1:
        win_len = 1

    accel_mag = _accel_magnitude(accel)

    feature_names = ["sma", "mean_accel", "var_accel", "dom_freq", "spec_energy"]
    features = []

    for start in range(0, n_samples - win_len + 1, win_len):
        end = start + win_len
        mag = accel_mag[start:end]

    
        sma = np.sum(np.abs(mag)) / len(mag)                      
        mean_accel = np.mean(mag)
        var_accel = np.var(mag)

        fft = np.fft.rfft(mag - np.mean(mag))
        freqs = np.fft.rfftfreq(len(mag), d=1.0 / fs)
        magnitude = np.abs(fft)

        if len(magnitude) > 1:
            dom_freq_idx = np.argmax(magnitude[1:]) + 1
            dom_freq = freqs[dom_freq_idx]
        else:
            dom_freq = 0.0

        spec_energy = np.sum(magnitude ** 2)

        features.append([sma, mean_accel, var_accel, dom_freq, spec_energy])

    return np.array(features), feature_names


def z_score_normalize(X, mean=None, std=None):
    """
    Z-score normalization: (X - mean) / std.

    Parameters
    ----------
    X : (n_windows, n_features) array
    mean : (n_features,) array or None — if None, computed from X
    std  : (n_features,) array or None — if None, computed from X

    Returns
    -------
    X_norm : (n_windows, n_features) array
    mean : (n_features,) array
    std  : (n_features,) array
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    std = np.where(std == 0, 1e-8, std)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def extract_all_features_from_dir(data_dir, window_sec=2.0):
    """
    Walk a directory of activity subfolders, extract features from every CSV.

    Returns
    -------
    X : (total_windows, n_features) array
    y : (total_windows,) array of activity label strings
    file_info : list of dicts with filepath, activity, n_windows
    """
    all_X = []
    all_y = []
    file_info = []

    for activity in sorted(os.listdir(data_dir)):
        activity_dir = os.path.join(data_dir, activity)
        if not os.path.isdir(activity_dir):
            continue
        for fname in sorted(os.listdir(activity_dir)):
            if not fname.endswith(".csv"):
                continue
            filepath = os.path.join(activity_dir, fname)
            timestamps, accel, gyro, fs, activity_label = read_csv(filepath)
            X_file, feature_names = extract_features(accel, gyro, fs, window_sec=window_sec)
            n_win = len(X_file)
            all_X.append(X_file)
            all_y.extend([activity_label] * n_win)
            file_info.append({
                "filepath": filepath,
                "activity": activity_label,
                "n_windows": n_win,
                "fs": fs,
            })

    X = np.vstack(all_X) if all_X else np.empty((0, 5))
    y = np.array(all_y)
    return X, y, file_info, feature_names


if __name__ == "__main__":
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
    X, y, info, names = extract_all_features_from_dir(TRAIN_DIR)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature names: {names}")
    print(f"Labels: {np.unique(y)}")
    print(f"Samples per activity: { {a: int(np.sum(y == a)) for a in np.unique(y)} }")