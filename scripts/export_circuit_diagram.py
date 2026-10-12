"""
Generates circuit diagram figure for VQC architecture.
"""
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from src.vqc_classifier import BiomedicalVQC

vqc = BiomedicalVQC(num_qubits=6, num_layers=2)
fig = vqc.circuit.draw(output='mpl', style='iqp')
plt.title("6-Qubit Biomedical Variational Quantum Classifier (Angle Encoding + Hardware-Efficient Ansatz)", fontsize=11, pad=12, fontweight='bold')
plt.tight_layout()
plt.savefig("results/figures/vqc_circuit_diagram.png", dpi=300, bbox_inches='tight')
plt.close()
print("Circuit diagram saved to results/figures/vqc_circuit_diagram.png")
