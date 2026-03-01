"""Dataset loaders with preprocessing for quantum classification.

Each loader returns (X_train, X_test, y_train, y_test) ready for use
with QuantumClassifier or directly with Qiskit ML.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def _prepare(
    data: np.ndarray,
    target: np.ndarray,
    n_qubits: int,
    test_size: float,
    seed: int,
    binary_labels: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared preprocessing: filter, split, standardize, PCA, scale to [-1, 1]."""

    if binary_labels is not None:
        mask = np.isin(target, binary_labels)
        data = data[mask]
        target = (target[mask] == binary_labels[1]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=seed, stratify=target
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if n_qubits < X_train.shape[1]:
        pca = PCA(n_components=n_qubits).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    combined = np.vstack([X_train, X_test])
    minmax = MinMaxScaler((-1, 1)).fit(combined)
    X_train = minmax.transform(X_train)
    X_test = minmax.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_quantum_iris(
    n_qubits: int = 2, test_size: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iris dataset (classes 0 vs 1) preprocessed for quantum classification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    data, target = load_iris(return_X_y=True)
    return _prepare(data, target, n_qubits, test_size, seed, binary_labels=(0, 1))


def load_quantum_breast_cancer(
    n_qubits: int = 2, test_size: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Breast cancer dataset preprocessed for quantum classification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    data, target = load_breast_cancer(return_X_y=True)
    return _prepare(data, target, n_qubits, test_size, seed)


def load_quantum_wine(
    n_qubits: int = 2, test_size: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wine dataset (classes 0 vs 1) preprocessed for quantum classification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    data, target = load_wine(return_X_y=True)
    return _prepare(data, target, n_qubits, test_size, seed, binary_labels=(0, 1))


def load_quantum_adhoc(
    n_qubits: int = 2, training_size: int = 20, test_size: int = 10, gap: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Qiskit's ad-hoc dataset designed to show quantum advantage.

    This synthetic dataset has a decision boundary that is hard to learn
    classically but maps naturally to a quantum feature space.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    from qiskit_machine_learning.datasets import ad_hoc_data

    train_features, train_labels, test_features, test_labels = ad_hoc_data(
        training_size=training_size, test_size=test_size, n=n_qubits, gap=gap
    )
    return train_features, test_features, train_labels, test_labels
