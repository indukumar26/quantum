"""
Hardware Processors & Architecture Transpilation Module
=======================================================
Implements the Universal Challenge Framework hardware specifications:
  - Processor A: Linear Chain Topology (6 qubits: 0 - 1 - 2 - 3 - 4 - 5)
  - Processor B: T-Topology / Heavy-Hex Branch (Qubit 2 as degree-3 branching hub)
  - Processor C (Phase 2 Designed): Biomedical-Optimized Chordal Ring Topology
    (Hexagonal ring + diameter cross-connections to eliminate SWAPs for circular & all-pair ansatz)

Metrics computed per processor:
  - 2-Qubit Entangling Gate (CX) Count
  - Transpiled Circuit Depth
  - SWAP Gates Incurred
  - Total Gate Count
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

def get_processor_a_coupling_map() -> CouplingMap:
    """Processor A: 6-qubit Linear Chain Topology (0-1-2-3-4-5)"""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    bidirectional = edges + [(v, u) for u, v in edges]
    return CouplingMap(bidirectional)

def get_processor_b_coupling_map() -> CouplingMap:
    """Processor B: 6-qubit T-Shaped / Heavy-Hex subsystem (Qubit 2 as branching junction)"""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]
    bidirectional = edges + [(v, u) for u, v in edges]
    return CouplingMap(bidirectional)

def get_processor_c_coupling_map() -> CouplingMap:
    """
    Phase 2 Processor C: Custom Biomedical-Optimized Chordal Ring Architecture
    Hexagonal perimeter ring + diameter chords:
    Ring: (0,1), (1,2), (2,3), (3,4), (4,5), (5,0)
    Chords: (0,3), (1,4), (2,5)
    Diameter = 1-2 hops; native support for cyclic entangling layers without SWAPs.
    """
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    chords = [(0, 3), (1, 4), (2, 5)]
    edges = ring + chords
    bidirectional = edges + [(v, u) for u, v in edges]
    return CouplingMap(bidirectional)

PROCESSOR_REGISTRY = {
    "Processor A (Linear Chain)": {
        "coupling_map": get_processor_a_coupling_map,
        "name": "Processor A",
        "topology": "Linear Chain (1D Lattice)",
        "description": "Standard 1D chain of 6 qubits. Suffers from high routing overhead for non-adjacent interactions."
    },
    "Processor B (T-Branch)": {
        "coupling_map": get_processor_b_coupling_map,
        "name": "Processor B",
        "topology": "T-Topology (Junction)",
        "description": "Branching topology with Qubit 2 as central node. Sub-optimal for periodic circular entangling patterns."
    },
    "Processor C (Phase 2 Chordal Ring)": {
        "coupling_map": get_processor_c_coupling_map,
        "name": "Processor C (Phase 2)",
        "topology": "Chordal Ring (Hexagon + Diameter Chords)",
        "description": "Custom architecture engineered to match biomedical VQC cyclic ansatz graphs, eliminating SWAPs and reducing depth."
    }
}

def analyze_transpiled_circuit(transpiled_circuit: QuantumCircuit) -> Dict[str, Any]:
    """Extracts architectural benchmark metrics from a transpiled quantum circuit."""
    ops = transpiled_circuit.count_ops()
    cx_count = ops.get('cx', 0)
    swap_count = ops.get('swap', 0)
    depth = transpiled_circuit.depth()
    total_ops = sum(ops.values())
    
    return {
        "cx_count": cx_count,
        "swap_count": swap_count,
        "depth": depth,
        "total_gates": total_ops,
        "operations": dict(ops)
    }

def benchmark_circuit_on_processors(circuit: QuantumCircuit, optimization_level: int = 3, seed_transpiler: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Transpiles the biomedical VQC circuit across Processors A, B, and Phase 2 Processor C.
    Returns comparison metrics.
    """
    results = {}
    basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
    
    for key, info in PROCESSOR_REGISTRY.items():
        cmap = info["coupling_map"]()
        transpiled = transpile(
            circuit,
            coupling_map=cmap,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler
        )
        metrics = analyze_transpiled_circuit(transpiled)
        metrics["topology"] = info["topology"]
        metrics["circuit_transpiled"] = transpiled
        results[key] = metrics
        
    return results

def plot_processor_coupling_maps(save_path: Optional[str] = None):
    """Generates comparative network graph visualizations of the three processor coupling maps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    topologies = [
        ("Processor A: Linear Chain", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0)}, "#3B82F6"),
        ("Processor B: T-Topology", [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)], {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (2, -1.2)}, "#F59E0B"),
        ("Processor C: Phase 2 Chordal Ring", 
         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)], 
         {0: (0, 1), 1: (0.866, 0.5), 2: (0.866, -0.5), 3: (0, -1), 4: (-0.866, -0.5), 5: (-0.866, 0.5)}, 
         "#10B981")
    ]
    
    for ax, (title, edges, pos, color) in zip(axes, topologies):
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edges_from(edges)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, node_size=800, edgecolors='#1E293B', linewidths=2.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_weight='bold', font_size=12)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#64748B', width=2.5, alpha=0.85)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
