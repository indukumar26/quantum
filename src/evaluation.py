"""
Evaluation & Visualization Module
=================================
Generates publication-quality charts and comprehensive metrics reports:
  - ECG Signal Morphology Plots (Normal Sinus Rhythm vs Abnormal Arrhythmia)
  - Feature Distribution and Quantum Angle Visualizations
  - Training Loss Curves
  - ROC Curves and Confusion Matrices (Highlighting Recall)
  - Hardware Architecture Metrics (CX Count, Depth, SWAP Count across Processors)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, Any, List, Optional

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_ecg_segments(raw_signals: np.ndarray, labels: np.ndarray, fs: float = 360.0, save_path: Optional[str] = None):
    """Plots comparative ECG waveforms: Normal Sinus Rhythm vs Arrhythmia."""
    norm_idx = np.where(labels == 0)[0][0]
    abnorm_idx = np.where(labels == 1)[0][0]
    
    t = np.linspace(0, len(raw_signals[0]) / fs, len(raw_signals[0]))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Normal ECG
    axes[0].plot(t, raw_signals[norm_idx], color='#2563EB', lw=1.8, label='Normal Sinus Rhythm (Class 0)')
    axes[0].set_title('Normal Sinus Rhythm (NSR) - Regular R-R & Sharp QRS Complex', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Abnormal ECG
    axes[1].plot(t, raw_signals[abnorm_idx], color='#DC2626', lw=1.8, label='Abnormal Arrhythmia / PVC (Class 1)')
    axes[1].set_title('Abnormal Cardiac Signal - Ectopic Beat, Widened QRS & Altered Repolarization', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude (mV)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_distributions(df: pd.DataFrame, feature_cols: List[str], save_path: Optional[str] = None):
    """Plots distribution of clinical features grouped by diagnostic class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    palette = {0: '#3B82F6', 1: '#EF4444'}
    class_names = {0: 'Normal', 1: 'Abnormal'}
    plot_df = df.copy()
    plot_df['Diagnosis'] = plot_df['label'].map(class_names)
    
    for i, col in enumerate(feature_cols):
        sns.boxplot(data=plot_df, x='Diagnosis', y=col, ax=axes[i], palette=['#3B82F6', '#EF4444'], width=0.45)
        sns.stripplot(data=plot_df, x='Diagnosis', y=col, ax=axes[i], color='black', alpha=0.25, jitter=0.2, size=3.5)
        clean_name = col.replace('_', ' ').title()
        axes[i].set_title(clean_name, fontweight='bold', fontsize=11)
        axes[i].set_xlabel('')
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(models_eval: Dict[str, Dict[str, Any]], y_test: np.ndarray, save_path: Optional[str] = None):
    """Plots combined ROC curves comparing Classical Baseline vs VQC."""
    plt.figure(figsize=(8, 6.5))
    colors = ['#2563EB', '#10B981', '#8B5CF6', '#F59E0B']
    
    for (name, res), color in zip(models_eval.items(), colors):
        y_prob = res['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.4, label=f"{name} (AUC = {roc_score:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontweight='bold')
    plt.title('Biomedical Arrhythmia Classifier: ROC Analysis', fontweight='bold', fontsize=13, pad=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, alpha=0.35)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(lr_res: Dict[str, Any], vqc_res: Dict[str, Any], save_path: Optional[str] = None):
    """Plots side-by-side confusion matrices emphasizing False Negatives (Recall)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, res, title in zip(axes, [lr_res, vqc_res], ['Logistic Regression Baseline', 'Variational Quantum Classifier (VQC)']):
        cm = np.array(res['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        rec = res['recall']
        f1 = res['f1_score']
        ax.set_title(f"{title}\nRecall: {rec:.2%} | F1: {f1:.3f}", fontweight='bold', fontsize=12, pad=10)
        ax.set_ylabel('True Clinical Diagnosis', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hardware_benchmarks(hw_metrics: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """Generates comparison bar chart of CX Count, Depth, and SWAP gates across Processors A, B, and C."""
    processors = list(hw_metrics.keys())
    short_names = ["Processor A\n(Linear)", "Processor B\n(T-Branch)", "Processor C\n(Phase 2 Ring)"]
    
    cx_counts = [hw_metrics[p]["cx_count"] for p in processors]
    depths = [hw_metrics[p]["depth"] for p in processors]
    swap_counts = [hw_metrics[p]["swap_count"] for p in processors]
    
    x = np.arange(len(processors))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width, cx_counts, width, label='2-Qubit CX Count', color='#3B82F6', edgecolor='#1E3A8A', lw=1.2)
    rects2 = ax.bar(x, depths, width, label='Circuit Depth', color='#F59E0B', edgecolor='#92400E', lw=1.2)
    rects3 = ax.bar(x + width, swap_counts, width, label='SWAP Gates Added', color='#EF4444', edgecolor='#7F1D1D', lw=1.2)
    
    ax.set_ylabel('Circuit Resource Count', fontweight='bold')
    ax.set_title('Universal Challenge Framework: Hardware Transpilation Benchmarks', fontweight='bold', fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, axis='y', alpha=0.35)
    
    # Value annotations on bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
