# Quantum Machine Learning Starter

Get started with Quantum Machine Learning in 5 lines of code. An sklearn-compatible wrapper around [Qiskit Machine Learning](https://github.com/qiskit-community/qiskit-machine-learning) that handles preprocessing, circuit construction, and benchmarking automatically.

## Quickstart

```python
from qml_starter import QuantumClassifier
from qml_starter.datasets import load_quantum_iris

X_train, X_test, y_train, y_test = load_quantum_iris(n_qubits=2)

clf = QuantumClassifier(n_qubits=2)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

## What This Does

Most QML tutorials are 50+ lines of boilerplate before you get a result. This package:

- **Reduces features to qubits automatically** — PCA + scaling to [-1, 1]
- **Builds quantum circuits for you** — ZZ feature map with configurable depth
- **Follows sklearn conventions** — `fit()`, `predict()`, `score()`
- **Compares quantum vs classical** — one-function benchmarking
- **Runs locally** — statevector simulator, no IBM account needed

## Installation

```bash
pip install qiskit qiskit-machine-learning scikit-learn
```

Then clone and install this package:

```bash
git clone https://github.com/dv-gorasiya/quantum-machine-learning.git
cd quantum-machine-learning
pip install -e .
```

## Usage

### Bring Your Own Data

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from qml_starter import QuantumClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = QuantumClassifier(n_qubits=4, feature_map_reps=2)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

`QuantumClassifier` takes raw data with any number of features and handles the reduction to `n_qubits` dimensions internally.

### Quantum vs Classical Comparison

```python
from qml_starter.benchmark import compare, print_comparison

results = compare(X_train, y_train, X_test, y_test, n_qubits=2)
print_comparison(results)
```

```
Classifier                  Accuracy       Time
-----------------------------------------------
Quantum SVM (QSVC)           0.9500      3.21s
Classical SVM (RBF)           0.9400      0.01s
Classical SVM (Linear)        0.9300      0.01s
```

### Variational Quantum Classifier

```python
clf = QuantumClassifier(n_qubits=2, method="vqc", optimizer_maxiter=100)
clf.fit(X_train, y_train)
```

### Pre-processed Datasets

```python
from qml_starter.datasets import (
    load_quantum_iris,
    load_quantum_breast_cancer,
    load_quantum_wine,
    load_quantum_adhoc,  # synthetic dataset designed for quantum advantage
)
```

### Visualization

```python
from qml_starter.plotting import plot_benchmark, plot_decision_regions

plot_benchmark(results)
plot_decision_regions(clf, X_test_2d, y_test)
```

Requires `matplotlib`: `pip install matplotlib`

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_qubits` | 2 | Number of qubits and PCA components |
| `method` | `"qsvc"` | `"qsvc"` (quantum kernel SVM) or `"vqc"` (variational) |
| `feature_map_reps` | 2 | ZZ feature map circuit depth |
| `entanglement` | `"linear"` | Qubit entanglement: `"linear"`, `"full"`, `"circular"` |
| `optimizer_maxiter` | 100 | Max optimizer iterations (VQC only) |
| `seed` | 42 | Random seed for reproducibility |
| `skip_preprocessing` | `False` | Skip PCA/scaling if data is already prepared |

## Requirements

- Python >= 3.10
- qiskit >= 2.0
- qiskit-machine-learning >= 0.8
- scikit-learn >= 1.2
- numpy >= 2.0

## Project Structure

```
qml_starter/
├── classifier.py    # QuantumClassifier — the main wrapper
├── datasets.py      # Pre-processed dataset loaders
├── benchmark.py     # Quantum vs classical comparison
└── plotting.py      # Visualization utilities
examples/
├── quickstart.py
├── custom_dataset.py
└── benchmark_comparison.py
tests/
└── test_classifier.py
```

## Background

Quantum kernel methods map classical data into a quantum feature space using parameterized circuits. For certain data distributions, this quantum feature space can capture patterns that are exponentially hard to compute classically. This package uses Qiskit's `ZZFeatureMap` and `FidelityQuantumKernel` to construct these mappings, wrapped in an interface familiar to anyone who has used scikit-learn.

**References:**

- Havlicek et al., [Supervised learning with quantum-enhanced feature spaces](https://arxiv.org/abs/1804.11326) (2019)
- [Qiskit Machine Learning documentation](https://qiskit-community.github.io/qiskit-machine-learning/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
