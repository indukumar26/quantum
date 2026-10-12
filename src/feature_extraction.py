"""
Biomedical Signal Processing & Feature Extraction Module
=========================================================
Extracts clinical and morphological features from short ECG segments (MIT-BIH Arrhythmia style).
Features extracted:
  1. heart_rate_bpm: Instantaneous heart rate in beats per minute
  2. rr_variability_sdnn: Beat-to-beat interval variability (SDNN in ms)
  3. qrs_peak_amplitude: Normalized R-peak amplitude
  4. qrs_duration_ms: QRS complex duration in milliseconds
  5. signal_energy: Total normalized signal power / energy
  6. spectral_power_ratio: Ratio of high-frequency (QRS 10-25 Hz) to low-frequency power
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List, Optional

def bandpass_filter(signal: np.ndarray, fs: float = 360.0, lowcut: float = 0.5, highcut: float = 40.0, order: int = 2) -> np.ndarray:
    """Applies a Butterworth bandpass filter to remove baseline wander and high-frequency noise."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_ecg_features(signal: np.ndarray, fs: float = 360.0) -> Dict[str, float]:
    """
    Extracts 6 diagnostic ECG features from a short signal segment (e.g., 2.0 - 3.0 seconds).
    
    Args:
        signal: 1D numpy array of ECG voltage amplitudes.
        fs: Sampling frequency in Hz (default: 360 Hz, MIT-BIH standard).
        
    Returns:
        Dictionary containing 6 clinical feature values.
    """
    filtered = bandpass_filter(signal, fs=fs)
    
    # R-peak detection
    # Distance between peaks minimum 0.2s (corresponds to max 300 bpm)
    min_dist = int(0.25 * fs)
    # Peak threshold: dynamic based on segment statistics
    height_thresh = np.mean(filtered) + 0.4 * np.std(filtered)
    peaks, properties = find_peaks(filtered, height=height_thresh, distance=min_dist)
    
    # Fallback if too few peaks detected in short window
    if len(peaks) < 2:
        # Relax threshold
        peaks, _ = find_peaks(filtered, distance=min_dist)
    
    # 1. Heart Rate (bpm)
    if len(peaks) >= 2:
        rr_intervals_sec = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals_sec)
        hr_bpm = 60.0 / mean_rr if mean_rr > 0 else 72.0
        # 2. RR Variability (SDNN in ms)
        sdnn_ms = np.std(rr_intervals_sec) * 1000.0 if len(rr_intervals_sec) > 1 else 35.0
    else:
        hr_bpm = 72.0
        sdnn_ms = 40.0
        
    # Clamp HR to physiological range
    hr_bpm = float(np.clip(hr_bpm, 40.0, 180.0))
    sdnn_ms = float(np.clip(sdnn_ms, 5.0, 250.0))
    
    # 3. Peak Amplitude (maximum absolute QRS deflection)
    peak_amp = float(np.max(np.abs(filtered)))
    
    # 4. QRS Duration Estimation (ms)
    # Measure width at half prominence of dominant peak
    if len(peaks) > 0:
        dominant_peak = peaks[np.argmax(np.abs(filtered[peaks]))]
        half_height = 0.5 * np.abs(filtered[dominant_peak])
        # Find left and right bounds where signal stays above half_height
        left_idx = dominant_peak
        while left_idx > 0 and np.abs(filtered[left_idx]) > half_height:
            left_idx -= 1
        right_idx = dominant_peak
        while right_idx < len(filtered) - 1 and np.abs(filtered[right_idx]) > half_height:
            right_idx += 1
        qrs_duration_ms = float((right_idx - left_idx) / fs * 1000.0)
    else:
        qrs_duration_ms = 95.0
    qrs_duration_ms = float(np.clip(qrs_duration_ms, 50.0, 240.0))
    
    # 5. Signal Energy (Normalized Power)
    signal_energy = float(np.sum(filtered ** 2) / len(filtered))
    
    # 6. Spectral Power Ratio (High-frequency QRS energy / low-frequency baseline energy)
    freqs, psd = welch(filtered, fs=fs, nperseg=min(len(filtered), 256))
    lf_mask = (freqs >= 0.5) & (freqs < 5.0)
    hf_mask = (freqs >= 10.0) & (freqs <= 25.0)
    lf_power = np.sum(psd[lf_mask]) + 1e-8
    hf_power = np.sum(psd[hf_mask]) + 1e-8
    spectral_ratio = float(hf_power / lf_power)
    
    return {
        "heart_rate_bpm": hr_bpm,
        "rr_variability_sdnn": sdnn_ms,
        "qrs_peak_amplitude": peak_amp,
        "qrs_duration_ms": qrs_duration_ms,
        "signal_energy": signal_energy,
        "spectral_power_ratio": spectral_ratio
    }

def synthesize_mitbih_ecg(label: int, duration_sec: float = 2.0, fs: float = 360.0, noise_std: float = 0.04, seed: Optional[int] = None) -> np.ndarray:
    """
    Synthesizes a realistic ECG segment matching MIT-BIH Arrhythmia characteristics.
    
    Label 0: Normal Sinus Rhythm (NSR)
      - Regular R-R interval (65-80 bpm), narrow QRS (~80-100 ms), standard P and T waves.
      
    Label 1: Abnormal / Arrhythmia (Premature Ventricular Contraction PVC / Ectopic)
      - Irregular timing, compensatory pause, bizarre widened QRS (>130 ms), high peak amplitude, inverted T wave.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, duration_sec, int(duration_sec * fs), endpoint=False)
    ecg = np.zeros_like(t)
    
    if label == 0:
        # Normal Sinus Rhythm
        bpm = np.random.uniform(62.0, 82.0)
        rr_sec = 60.0 / bpm
        num_beats = int(np.ceil(duration_sec / rr_sec)) + 1
        beat_starts = [0.1 + i * rr_sec + np.random.normal(0, 0.015) for i in range(num_beats)]
        
        for t0 in beat_starts:
            # P wave (at t0 - 0.16s)
            ecg += 0.18 * np.exp(-((t - (t0 - 0.16)) / 0.035)**2)
            # Q wave
            ecg -= 0.15 * np.exp(-((t - (t0 - 0.04)) / 0.015)**2)
            # R peak (sharp, narrow)
            ecg += 1.25 * np.exp(-((t - t0) / 0.022)**2)
            # S wave
            ecg -= 0.25 * np.exp(-((t - (t0 + 0.04)) / 0.018)**2)
            # T wave
            ecg += 0.32 * np.exp(-((t - (t0 + 0.22)) / 0.065)**2)
            
    else:
        # Abnormal / PVC / Arrhythmia
        # High beat-to-beat irregularity or early ectopic beat with widened QRS
        bpm = np.random.uniform(75.0, 115.0)
        t_ectopic = np.random.uniform(0.6, 1.2)
        beat_starts = [0.15, t_ectopic, t_ectopic + 0.9] # Early beat followed by compensatory pause
        
        for idx, t0 in enumerate(beat_starts):
            if idx == 1:
                # Ectopic PVC beat: no P wave, widened high-amplitude deformed QRS, inverted T wave
                ecg += 1.85 * np.exp(-((t - t0) / 0.065)**2) # Widened QRS (>130 ms)
                ecg -= 0.65 * np.exp(-((t - (t0 + 0.08)) / 0.05)**2) # Deep S wave
                ecg -= 0.45 * np.exp(-((t - (t0 + 0.28)) / 0.09)**2) # Inverted T wave
            else:
                ecg += 1.10 * np.exp(-((t - t0) / 0.025)**2)
                ecg += 0.25 * np.exp(-((t - (t0 + 0.20)) / 0.06)**2)
                
    # Add realistic baseline wander (low freq 0.15 Hz respiration) and high freq sensor noise
    baseline_wander = 0.08 * np.sin(2 * np.pi * 0.25 * t + np.random.uniform(0, 2*np.pi))
    sensor_noise = np.random.normal(0, noise_std, size=len(t))
    ecg += baseline_wander + sensor_noise
    return ecg

def generate_benchmark_dataset(num_samples: int = 350, abnormal_ratio: float = 0.35, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a calibrated benchmark dataset of 350 ECG signal segments (200-500 range)
    with fixed labels and clinical features extracted.
    
    Args:
        num_samples: Number of signal segments (e.g. 350).
        abnormal_ratio: Proportion of abnormal / arrhythmia cases (default 0.35, reflects real-world clinical imbalance).
        seed: Random seed for exact reproducibility.
        
    Returns:
        Tuple of (DataFrame of extracted features and labels, 2D array of raw signals)
    """
    np.random.seed(seed)
    num_abnormal = int(num_samples * abnormal_ratio)
    num_normal = num_samples - num_abnormal
    
    labels = np.array([0] * num_normal + [1] * num_abnormal)
    indices = np.random.permutation(num_samples)
    labels = labels[indices]
    
    feature_rows = []
    raw_signals = []
    
    for i, label in enumerate(labels):
        sig = synthesize_mitbih_ecg(label=label, seed=seed + i)
        raw_signals.append(sig)
        feats = extract_ecg_features(sig)
        feats["label"] = label
        feats["sample_id"] = f"ECG_{i:04d}"
        feature_rows.append(feats)
        
    df = pd.DataFrame(feature_rows)
    return df, np.array(raw_signals)

def scale_features_to_quantum_angle(X: np.ndarray, feature_range: Tuple[float, float] = (0.0, np.pi)) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalizes numerical biomedical features to the quantum rotation angle interval [0, pi]
    as required for angle encoding: theta_i in [0, pi].
    
    Args:
        X: (N, D) numeric feature matrix.
        feature_range: Target range, default (0.0, np.pi).
        
    Returns:
        X_scaled: Transformed feature matrix in [0, pi].
        scaler: Fitted scikit-learn MinMaxScaler instance.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
