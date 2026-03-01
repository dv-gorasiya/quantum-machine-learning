"""5-line quickstart: quantum classification on the Iris dataset."""

from qml_starter import QuantumClassifier
from qml_starter.datasets import load_quantum_iris

X_train, X_test, y_train, y_test = load_quantum_iris(n_qubits=2)

clf = QuantumClassifier(n_qubits=2)
clf.fit(X_train, y_train)

print(f"Quantum SVM accuracy: {clf.score(X_test, y_test):.2f}")
