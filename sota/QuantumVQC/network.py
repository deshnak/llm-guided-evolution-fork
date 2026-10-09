"""Variational quantum classifier for the UCI Seeds dataset.

Everything above the first option marker is fixed: data loading, forward
simulation, the training budget, scoring and result writing. The blocks
below it define the circuit and the training procedure.
"""
import json
import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

N_FEATURES = 7
N_QUBITS = 7
N_CLASSES = 3
SPLIT_SEED = 42
START_SEEDS = [0, 1, 2, 3, 4]
# Training budget per start, in full passes over the training set
PASS_BUDGET = 178

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seeds_dataset.txt")


def load_data(data_path=DEFAULT_DATA):
    data = np.genfromtxt(data_path)
    assert data.shape == (210, N_FEATURES + 1), f"unexpected Seeds shape {data.shape}"
    X = data[:, :N_FEATURES]
    y = data[:, N_FEATURES].astype(int) - 1
    # Stratified 60/20/20
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SPLIT_SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SPLIT_SEED)
    # Angles in [0, pi], scaler fit on train only
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train)
    return {
        "train": (scaler.transform(X_train), y_train),
        "val": (scaler.transform(X_val), y_val),
        "test": (scaler.transform(X_test), y_test),
    }


class BudgetExhausted(Exception):
    pass


class Budget:
    """Counts samples sent through forward() during training."""

    def __init__(self, n_train):
        self.n_train = n_train
        self.limit = PASS_BUDGET * n_train
        self.used = 0

    def charge(self, n_samples):
        if self.used + n_samples > self.limit:
            raise BudgetExhausted()
        self.used += n_samples

    @property
    def passes(self):
        return self.used / self.n_train


def forward(model, theta, X, budget=None):
    """Class probabilities for each row of X, shape (len(X), N_CLASSES)."""
    qc, x_params, theta_params = model
    if budget is not None:
        budget.charge(len(X))
    bound = qc.assign_parameters(dict(zip(theta_params, theta)))
    out = np.empty((len(X), N_CLASSES))
    for i, x in enumerate(X):
        sv = Statevector(bound.assign_parameters(dict(zip(x_params, x))))
        p = np.clip(np.asarray(readout(sv, N_CLASSES), dtype=float)[:N_CLASSES], 1e-12, None)
        out[i] = p / p.sum()
    return out


def cross_entropy(probs, y):
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


def accuracy(probs, y):
    return float(np.mean(np.argmax(probs, axis=1) == y))


def flatten_circuit(qc):
    """Decompose until only standard gates remain."""
    standard = set(get_standard_gate_name_mapping())
    for _ in range(10):
        if all(inst.operation.name in standard | {"barrier", "measure"} for inst in qc.data):
            break
        qc = qc.decompose()
    return qc


def circuit_metrics(model):
    qc, x_params, _ = model
    qc = flatten_circuit(qc)
    x_set = set(x_params)
    gates = [inst for inst in qc.data if inst.operation.name not in ("barrier", "measure")]

    def is_encoding(inst):
        return any(hasattr(p, "parameters") and p.parameters & x_set for p in inst.operation.params)

    return {
        "gate_count": len(gates),
        "gate_count_no_encoding": sum(1 for g in gates if not is_encoding(g)),
        "encoding_gate_count": sum(1 for g in gates if is_encoding(g)),
        "two_qubit_gate_count": sum(1 for g in gates if len(g.qubits) == 2),
        "depth": qc.depth(),
        "n_params": len(model[2]),
    }


def train_one_start(model, data, seed):
    X_train, y_train = data["train"]
    budget = Budget(len(X_train))
    best = {"loss": np.inf, "theta": None}

    def cost_fn(theta):
        loss = cross_entropy(forward(model, theta, X_train, budget), y_train)
        if loss < best["loss"]:
            best["loss"], best["theta"] = loss, np.array(theta, dtype=float)
        return loss

    rng = np.random.default_rng(seed)
    try:
        theta = np.asarray(train(cost_fn, len(model[2]), rng, PASS_BUDGET), dtype=float)
        if theta.shape != (len(model[2]),):
            theta = best["theta"]
    except BudgetExhausted:
        theta = best["theta"]
    if theta is None:
        raise RuntimeError("training never evaluated the cost function")

    result = {"seed": seed, "passes_used": budget.passes}
    for split, (X, y) in data.items():
        probs = forward(model, theta, X)
        result[f"{split}_acc"] = accuracy(probs, y)
        result[f"{split}_ce"] = cross_entropy(probs, y)
    return result


def write_results(gene_id, metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{gene_id}_results.csv"), "w") as f:
        f.write(f"{metrics['obj1']}, {metrics['obj2']}")
    with open(os.path.join(results_dir, f"{gene_id}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main(gene_id="seed", data_path=DEFAULT_DATA, results_dir="results"):
    data = load_data(data_path)
    model = build_circuit(N_QUBITS)
    assert len(model[1]) == N_FEATURES, "circuit must take one input parameter per feature"

    starts = [train_one_start(model, data, s) for s in START_SEEDS]
    metrics = {"gene_id": gene_id, "pass_budget": PASS_BUDGET, "start_seeds": START_SEEDS}
    for key in [k for k in starts[0] if k != "seed"]:
        vals = [s[key] for s in starts]
        metrics[f"{key}_mean"] = float(np.mean(vals))
        metrics[f"{key}_std"] = float(np.std(vals))
    metrics.update(circuit_metrics(model))
    metrics["per_start"] = starts
    # Selection objectives (both minimized). Test metrics are never used here.
    metrics["obj1"] = 1.0 - metrics["val_acc_mean"]
    metrics["obj2"] = metrics["gate_count"]
    write_results(gene_id, metrics, results_dir)
    return metrics


# --OPTION--
def feature_map(qc, x):
    """Angle encoding: one RY rotation per feature."""
    for i in range(qc.num_qubits):
        qc.ry(x[i], i)


# --OPTION--
def n_layer_params(n_qubits):
    return 2 * n_qubits


def variational_layer(qc, params):
    """Trainable RY then RZ on every qubit."""
    for i in range(qc.num_qubits):
        qc.ry(params[2 * i], i)
        qc.rz(params[2 * i + 1], i)


# --OPTION--
def entangle(qc):
    """CNOT ring."""
    n = qc.num_qubits
    for i in range(n):
        qc.cx(i, (i + 1) % n)


# --OPTION--
N_LAYERS = 2


def build_circuit(n_qubits):
    x = ParameterVector("x", n_qubits)
    per_layer = n_layer_params(n_qubits)
    theta = ParameterVector("theta", N_LAYERS * per_layer)
    qc = QuantumCircuit(n_qubits)
    feature_map(qc, x)
    for layer in range(N_LAYERS):
        variational_layer(qc, theta[layer * per_layer:(layer + 1) * per_layer])
        entangle(qc)
    return qc, list(x), list(theta)


# --OPTION--
def readout(sv, n_classes):
    """Marginal over qubits 0 and 1, first n_classes basis states, renormalized."""
    p = sv.probabilities([0, 1])[:n_classes]
    return p / p.sum()


# --OPTION--
def train(cost_fn, n_params, rng, max_passes):
    theta0 = rng.uniform(0, 2 * np.pi, n_params)
    res = minimize(cost_fn, theta0, method="COBYLA", options={"maxiter": max_passes})
    return res.x


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
