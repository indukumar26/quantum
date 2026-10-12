"""
Variational Quantum Classifier (VQC) Module
===========================================
Implements a 6-qubit Hybrid Quantum-Classical Variational Classifier for ECG screening:
  1. Angle Encoding: Encodes 6 normalized features into rotation angles theta in [0, pi].
  2. Parameterized Ansatz: Multi-layer RealAmplitudes / Hardware-Efficient entangling circuit.
  3. Expectation Value Measurement: Pauli-Z observable <Z_0>.
  4. Class-Weighted Loss: Prioritizes Recall to penalize false negatives on abnormal cases.
  5. Optimization: Classical gradient-based / COBYLA / Adam optimizer.
  6. Direct compatibility with Qiskit 2.x and hardware transpilation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix

class BiomedicalVQC:
    """
    6-Qubit Variational Quantum Classifier tailored for biomedical arrhythmia detection.
    """
    def __init__(self, num_qubits: int = 6, num_layers: int = 2, learning_rate: float = 0.05, max_iter: int = 60, seed: int = 42):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.seed = seed
        np.random.seed(seed)
        
        # Build parameterized template circuit
        self.circuit, self.feature_params, self.var_params = self._build_circuit()
        self.num_weights = len(self.var_params)
        self.weights = np.random.uniform(-0.5, 0.5, size=self.num_weights)
        self.bias = 0.0
        
        # Pauli Z on qubit 0 as readout observable
        # In Qiskit, qubit 0 corresponds to rightmost index in Pauli string
        pauli_str = "I" * (self.num_qubits - 1) + "Z"
        self.observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
        
        self.training_loss_history: List[float] = []

    def _build_circuit(self) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        """Constructs the parameterized VQC with angle encoding and cyclic entanglement."""
        x_params = ParameterVector('x', self.num_qubits)
        theta_params = ParameterVector('theta', self.num_layers * self.num_qubits * 2)
        
        qc = QuantumCircuit(self.num_qubits)
        
        # Layer 1: Angle Encoding (Ry rotations)
        for i in range(self.num_qubits):
            qc.ry(x_params[i], i)
            
        param_idx = 0
        # Variational Layers
        for l in range(self.num_layers):
            qc.barrier()
            # Parameterized Single-Qubit Rotations
            for i in range(self.num_qubits):
                qc.ry(theta_params[param_idx], i)
                param_idx += 1
                qc.rz(theta_params[param_idx], i)
                param_idx += 1
                
            # Cyclic Entanglement (Ring topology: 0->1->2->3->4->5->0)
            for i in range(self.num_qubits):
                qc.cx(i, (i + 1) % self.num_qubits)
                
        return qc, x_params, theta_params

    def get_circuit_for_sample(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> QuantumCircuit:
        """Returns a bound QuantumCircuit for a specific sample and weight configuration."""
        if weights is None:
            weights = self.weights
        param_dict = {}
        for p, val in zip(self.feature_params, x):
            param_dict[p] = float(val)
        for p, val in zip(self.var_params, weights):
            param_dict[p] = float(val)
            
        return self.circuit.assign_parameters(param_dict)

    def _compute_expectation(self, bound_qc: QuantumCircuit) -> float:
        """Computes <Psi | Z_0 | Psi> using fast exact statevector simulation."""
        sv = Statevector.from_instruction(bound_qc)
        exp_val = sv.expectation_value(self.observable).real
        return float(exp_val)

    def predict_sample_proba(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Outputs predicted probability P(abnormal=1) in [0, 1]."""
        bound_qc = self.get_circuit_for_sample(x, weights)
        exp_val = self._compute_expectation(bound_qc)
        # Shift and map expectation [-1, 1] through sigmoid: prob = 1 / (1 + exp(- (w*exp + b)))
        z = 2.0 * exp_val + self.bias
        prob = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(prob, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes predicted probabilities for batch of samples."""
        probs = [self.predict_sample_proba(x, weights) for x in X]
        return np.array(probs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class labels (0: Normal, 1: Abnormal)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def compute_weighted_loss(self, y_true: np.ndarray, y_pred_prob: np.ndarray, w0: float, w1: float) -> float:
        """Computes binary cross-entropy with clinical class weights (prioritizing recall)."""
        eps = 1e-7
        p = np.clip(y_pred_prob, eps, 1.0 - eps)
        loss = -np.mean(w1 * y_true * np.log(p) + w0 * (1.0 - y_true) * np.log(1.0 - p))
        return float(loss)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> 'BiomedicalVQC':
        """
        Trains the VQC parameters using Adam optimizer with Parameter Shift / Finite Difference gradients
        and class-weighted loss.
        """
        n_samples = len(y_train)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        
        # Clinical Class Weights: w1 higher to penalize missed abnormalities (Recall priority)
        w0 = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
        w1 = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
        
        # Adam Optimizer Variables
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Subsample for efficient stochastic mini-batching
        batch_size = min(32, n_samples)
        
        if verbose:
            print(f"[VQC Training] Starting training: {self.max_iter} iterations, batch size {batch_size}, w0={w0:.2f}, w1={w1:.2f}")

        shift = np.pi / 2.0
        
        for iteration in range(1, self.max_iter + 1):
            # Sample mini-batch
            batch_idx = np.random.choice(n_samples, size=batch_size, replace=False)
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx]
            
            # Current probabilities & loss
            curr_probs = np.array([self.predict_sample_proba(x) for x in X_b])
            curr_loss = self.compute_weighted_loss(y_b, curr_probs, w0, w1)
            self.training_loss_history.append(curr_loss)
            
            # Compute gradient for weights using parameter-shift rule / central difference
            grad_w = np.zeros_like(self.weights)
            
            # Select subset of active weights per step for fast convergence
            active_indices = np.random.choice(self.num_weights, size=min(8, self.num_weights), replace=False)
            for idx in active_indices:
                w_plus = self.weights.copy()
                w_plus[idx] += shift
                p_plus = np.array([self.predict_sample_proba(x, weights=w_plus) for x in X_b])
                loss_plus = self.compute_weighted_loss(y_b, p_plus, w0, w1)
                
                w_minus = self.weights.copy()
                w_minus[idx] -= shift
                p_minus = np.array([self.predict_sample_proba(x, weights=w_minus) for x in X_b])
                loss_minus = self.compute_weighted_loss(y_b, p_minus, w0, w1)
                
                grad_w[idx] = (loss_plus - loss_minus) / (2.0 * np.sin(shift))
                
            # Bias gradient
            grad_bias = np.mean((curr_probs - y_b) * (w1 * y_b + w0 * (1 - y_b)))
            
            # Adam update
            m = beta1 * m + (1.0 - beta1) * grad_w
            v = beta2 * v + (1.0 - beta2) * (grad_w ** 2)
            m_hat = m / (1.0 - beta1 ** iteration)
            v_hat = v / (1.0 - beta2 ** iteration)
            
            self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            self.bias -= self.learning_rate * grad_bias
            
            if verbose and (iteration % 10 == 0 or iteration == 1):
                print(f"  Iteration {iteration:02d}/{self.max_iter:02d} - Weighted Loss: {curr_loss:.4f}")
                
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates the trained VQC on test data, computing clinical and ML metrics."""
        probs = self.predict_proba(X_test)
        preds = (probs >= 0.5).astype(int)
        
        acc = float(accuracy_score(y_test, preds))
        rec = float(recall_score(y_test, preds, zero_division=0)) # Primary Metric
        f1 = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5
        cm = confusion_matrix(y_test, preds).tolist()
        
        return {
            "accuracy": acc,
            "recall": rec,          # Primary Metric
            "f1_score": f1,
            "precision": prec,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "y_pred": preds,
            "y_prob": probs
        }
