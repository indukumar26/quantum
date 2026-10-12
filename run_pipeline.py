"""
End-to-End Execution Script: PS-Q1 Biomedical Signal Classifier
==============================================================
Runs the full scientific and hardware benchmark pipeline:
  1. Generates/loads 350-sample MIT-BIH biomedical ECG dataset (200-500 samples specification).
  2. Extracts 6 diagnostic clinical features and normalizes to quantum angles [0, pi].
  3. Trains classical baselines (Logistic Regression with balanced class weights, RF, SVC).
  4. Builds and trains 6-qubit Biomedical VQC with class-weighted loss (prioritizing Recall).
  5. Evaluates and compares ML & clinical screening metrics (Recall, F1, Accuracy, Precision, AUC).
  6. Transpiles VQC circuit across Processor A, Processor B, and Phase 2 Processor C.
  7. Computes hardware metrics (CX count, depth, SWAP gates).
  8. Saves all artifacts, figures, and benchmark tables.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_extraction import (
    generate_benchmark_dataset,
    scale_features_to_quantum_angle
)
from src.classical_baseline import ClassicalBaselines
from src.vqc_classifier import BiomedicalVQC
from src.hardware_processors import (
    benchmark_circuit_on_processors,
    plot_processor_coupling_maps
)
from src.evaluation import (
    plot_ecg_segments,
    plot_feature_distributions,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_hardware_benchmarks
)

def main():
    print("=" * 75)
    print("PS-Q1: Biomedical Signal Classifier - Universal Challenge Framework")
    print("=" * 75)
    
    # 0. Setup directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Dataset Generation (MIT-BIH ECG Segments)
    print("\n[Step 1] Synthesizing & Extracting MIT-BIH Arrhythmia ECG Features (N=350)...")
    df_dataset, raw_signals = generate_benchmark_dataset(num_samples=350, abnormal_ratio=0.35, seed=42)
    
    # Save dataset to disk
    dataset_path = "data/mitbih_biomedical_dataset.csv"
    df_dataset.to_csv(dataset_path, index=False)
    print(f"  Dataset saved to: {dataset_path}")
    print(f"  Total samples: {len(df_dataset)} (Normal: {sum(df_dataset['label']==0)}, Abnormal: {sum(df_dataset['label']==1)})")
    
    feature_cols = [
        "heart_rate_bpm",
        "rr_variability_sdnn",
        "qrs_peak_amplitude",
        "qrs_duration_ms",
        "signal_energy",
        "spectral_power_ratio"
    ]
    
    # Generate ECG waveform plots and feature distribution plots
    plot_ecg_segments(raw_signals, df_dataset['label'].values, fs=360.0, save_path="results/figures/ecg_samples_normal_abnormal.png")
    plot_feature_distributions(df_dataset, feature_cols, save_path="results/figures/feature_distributions.png")
    print("  Saved ECG signal & feature distribution plots.")
    
    # 2. Angle Normalization to [0, pi] and Fixed Train/Test Split
    print("\n[Step 2] Normalizing features to quantum angle domain [0, pi] & Stratified Split...")
    X_raw = df_dataset[feature_cols].values
    y = df_dataset['label'].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Normalize features to [0, pi]
    X_train_quantum, scaler = scale_features_to_quantum_angle(X_train_raw, feature_range=(0.0, np.pi))
    X_test_quantum = scaler.transform(X_test_raw)
    
    print(f"  Train set: {len(y_train)} samples | Test set: {len(y_test)} samples")
    print(f"  Feature bounds: min={X_train_quantum.min():.3f}, max={X_train_quantum.max():.3f} rad (in [0, pi])")
    
    # 3. Train Classical Baselines
    print("\n[Step 3] Training Classical ML Baselines (Logistic Regression with class_weight='balanced')...")
    baselines = ClassicalBaselines(random_state=42)
    classical_results = baselines.fit_and_evaluate(X_train_quantum, y_train, X_test_quantum, y_test)
    
    for name, res in classical_results.items():
        print(f"  {name:32s} | Recall: {res['recall']:.2%} | F1: {res['f1_score']:.3f} | Accuracy: {res['accuracy']:.2%} | AUC: {res['roc_auc']:.3f}")
        
    # 4. Train Variational Quantum Classifier (VQC)
    print("\n[Step 4] Constructing & Training 6-Qubit Hybrid Variational Quantum Classifier (VQC)...")
    vqc = BiomedicalVQC(num_qubits=6, num_layers=2, learning_rate=0.06, max_iter=50, seed=42)
    vqc.fit(X_train_quantum, y_train, verbose=True)
    vqc_results = vqc.evaluate(X_test_quantum, y_test)
    
    print(f"\n  [VQC Performance] Recall: {vqc_results['recall']:.2%} (Primary Metric) | F1: {vqc_results['f1_score']:.3f} | Accuracy: {vqc_results['accuracy']:.2%} | AUC: {vqc_results['roc_auc']:.3f}")
    
    # 5. Combined Model Comparison Visualizations
    all_models_eval = {
        "Logistic Regression (Baseline)": classical_results["Logistic Regression"],
        "Support Vector Classifier": classical_results["Support Vector Classifier (RBF)"],
        "Random Forest": classical_results["Random Forest"],
        "Hybrid VQC (6 Qubits)": vqc_results
    }
    
    plot_roc_curves(all_models_eval, y_test, save_path="results/figures/roc_curves.png")
    plot_confusion_matrices(classical_results["Logistic Regression"], vqc_results, save_path="results/figures/confusion_matrices.png")
    
    # Save training loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(vqc.training_loss_history) + 1), vqc.training_loss_history, color='#2563EB', lw=2.2)
    plt.xlabel('Optimization Iteration', fontweight='bold')
    plt.ylabel('Class-Weighted Cross Entropy', fontweight='bold')
    plt.title('VQC Convergence with Recall-Prioritizing Class Weights', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/loss_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved ROC curves, confusion matrices, and loss convergence plots.")
    
    # 6. Hardware Transpilation & Phase 2 Processor C Benchmarking
    print("\n[Step 5] Benchmarking Quantum Circuit across Processors A, B, and Phase 2 Processor C...")
    # Create representative circuit instance
    sample_circuit = vqc.get_circuit_for_sample(X_test_quantum[0])
    
    # Plot coupling maps
    plot_processor_coupling_maps(save_path="results/figures/processor_coupling_maps.png")
    
    # Run transpilation benchmarks
    hw_benchmarks = benchmark_circuit_on_processors(sample_circuit, optimization_level=3, seed_transpiler=42)
    
    # Clean output for printing and JSON
    clean_hw = {}
    print("\n" + "=" * 80)
    print(f"{'Processor':35s} | {'CX Count':10s} | {'Circuit Depth':14s} | {'SWAP Count':10s}")
    print("-" * 80)
    for p_name, data in hw_benchmarks.items():
        print(f"{p_name:35s} | {data['cx_count']:10d} | {data['depth']:14d} | {data['swap_count']:10d}")
        clean_hw[p_name] = {
            "topology": data["topology"],
            "cx_count": data["cx_count"],
            "depth": data["depth"],
            "swap_count": data["swap_count"],
            "total_gates": data["total_gates"]
        }
    print("=" * 80)
    
    plot_hardware_benchmarks(hw_benchmarks, save_path="results/figures/hardware_tradeoffs_comparison.png")
    print("  Saved coupling maps and hardware tradeoff comparison plots.")
    
    # 7. Save Comprehensive JSON Summary
    summary = {
        "dataset": {
            "total_samples": len(df_dataset),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "normal_samples": int(sum(y == 0)),
            "abnormal_samples": int(sum(y == 1)),
            "features": feature_cols
        },
        "ml_performance": {
            "Logistic Regression": {k: v for k, v in classical_results["Logistic Regression"].items() if k not in ["y_pred", "y_prob"]},
            "Support Vector Classifier": {k: v for k, v in classical_results["Support Vector Classifier (RBF)"].items() if k not in ["y_pred", "y_prob"]},
            "Random Forest": {k: v for k, v in classical_results["Random Forest"].items() if k not in ["y_pred", "y_prob"]},
            "Biomedical VQC": {k: v for k, v in vqc_results.items() if k not in ["y_pred", "y_prob"]}
        },
        "hardware_benchmarks": clean_hw
    }
    
    with open("results/benchmark_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Done] Pipeline executed successfully. Results written to results/benchmark_metrics.json")

if __name__ == "__main__":
    main()
