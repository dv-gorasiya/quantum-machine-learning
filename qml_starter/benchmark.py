"""Compare quantum and classical classifiers on the same data."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC

from qml_starter.classifier import QuantumClassifier


@dataclass
class BenchmarkResult:
    """Stores accuracy and timing for a single classifier run."""

    name: str
    accuracy: float
    train_time_seconds: float

    def __repr__(self) -> str:
        return f"{self.name:<25} accuracy={self.accuracy:.4f}  time={self.train_time_seconds:.2f}s"


def compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = 2,
    feature_map_reps: int = 2,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run quantum SVM, classical RBF SVM, and linear SVM on the same data.

    Data should be raw (unscaled) — QuantumClassifier handles its own
    preprocessing. Classical SVMs receive the same raw data.

    Parameters
    ----------
    X_train, y_train : training data and labels
    X_test, y_test : test data and labels
    n_qubits : number of qubits for the quantum classifier
    feature_map_reps : ZZ feature map repetitions
    seed : random seed

    Returns
    -------
    list of BenchmarkResult, sorted by accuracy descending
    """
    results = []

    # Quantum SVM (kernel method)
    t0 = time.time()
    qclf = QuantumClassifier(
        n_qubits=n_qubits,
        method="qsvc",
        feature_map_reps=feature_map_reps,
        seed=seed,
    )
    qclf.fit(X_train, y_train)
    q_acc = qclf.score(X_test, y_test)
    results.append(BenchmarkResult("Quantum SVM (QSVC)", q_acc, time.time() - t0))

    # Classical SVM — RBF kernel
    t0 = time.time()
    rbf = SVC(kernel="rbf", random_state=seed)
    rbf.fit(X_train, y_train)
    rbf_acc = rbf.score(X_test, y_test)
    results.append(BenchmarkResult("Classical SVM (RBF)", rbf_acc, time.time() - t0))

    # Classical SVM — linear kernel
    t0 = time.time()
    lin = SVC(kernel="linear", random_state=seed)
    lin.fit(X_train, y_train)
    lin_acc = lin.score(X_test, y_test)
    results.append(BenchmarkResult("Classical SVM (Linear)", lin_acc, time.time() - t0))

    results.sort(key=lambda r: r.accuracy, reverse=True)
    return results


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'Classifier':<25} {'Accuracy':>10} {'Time':>10}")
    print("-" * 47)
    for r in results:
        print(f"{r.name:<25} {r.accuracy:>10.4f} {r.train_time_seconds:>9.2f}s")
    print()
