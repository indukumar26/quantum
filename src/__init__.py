"""
PS-Q1 Biomedical Signal Classifier Package
Universal Challenge Framework
"""

from .feature_extraction import extract_ecg_features, synthesize_mitbih_ecg, generate_benchmark_dataset, scale_features_to_quantum_angle
from .vqc_classifier import BiomedicalVQC
from .classical_baseline import ClassicalBaselines
from .hardware_processors import (
    get_processor_a_coupling_map,
    get_processor_b_coupling_map,
    get_processor_c_coupling_map,
    benchmark_circuit_on_processors,
    plot_processor_coupling_maps,
    PROCESSOR_REGISTRY
)
from .evaluation import (
    plot_ecg_segments,
    plot_feature_distributions,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_hardware_benchmarks
)

__all__ = [
    "extract_ecg_features",
    "synthesize_mitbih_ecg",
    "generate_benchmark_dataset",
    "scale_features_to_quantum_angle",
    "BiomedicalVQC",
    "ClassicalBaselines",
    "get_processor_a_coupling_map",
    "get_processor_b_coupling_map",
    "get_processor_c_coupling_map",
    "benchmark_circuit_on_processors",
    "plot_processor_coupling_maps",
    "PROCESSOR_REGISTRY",
    "plot_ecg_segments",
    "plot_feature_distributions",
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_hardware_benchmarks"
]
