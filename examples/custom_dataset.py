"""Bring your own dataset — QuantumClassifier handles preprocessing."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from qml_starter import QuantumClassifier

# Generate a synthetic dataset with 20 features
X, y = make_classification(
    n_samples=200, n_features=20, n_informative=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# QuantumClassifier automatically reduces 20 features → 2 qubits via PCA,
# scales to [-1, 1], and builds the quantum kernel.
clf = QuantumClassifier(n_qubits=2, method="qsvc")
clf.fit(X_train, y_train)

print(f"Accuracy on 20-feature dataset (reduced to 2 qubits): {clf.score(X_test, y_test):.2f}")

# Try the variational approach instead
vqc = QuantumClassifier(n_qubits=2, method="vqc", optimizer_maxiter=50)
vqc.fit(X_train, y_train)

print(f"VQC accuracy: {vqc.score(X_test, y_test):.2f}")
