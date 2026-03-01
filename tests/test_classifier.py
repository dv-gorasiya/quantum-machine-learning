"""Tests for QuantumClassifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from qml_starter.classifier import QuantumClassifier


@pytest.fixture
def binary_dataset():
    X, y = make_classification(
        n_samples=40, n_features=4, n_informative=2, random_state=42
    )
    return X[:30], X[30:], y[:30], y[30:]


class TestQuantumClassifier:
    def test_fit_predict_qsvc(self, binary_dataset):
        X_train, X_test, y_train, _ = binary_dataset
        clf = QuantumClassifier(n_qubits=2, method="qsvc", seed=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_score_returns_float(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        clf = QuantumClassifier(n_qubits=2, seed=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_invalid_method_raises(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        clf = QuantumClassifier(method="invalid")
        with pytest.raises(ValueError, match="method must be"):
            clf.fit(X_train, y_train)

    def test_too_many_qubits_raises(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        clf = QuantumClassifier(n_qubits=100)
        with pytest.raises(ValueError, match="cannot exceed"):
            clf.fit(X_train, y_train)

    def test_predict_before_fit_raises(self):
        clf = QuantumClassifier()
        with pytest.raises(Exception):
            clf.predict(np.array([[1, 2]]))

    def test_skip_preprocessing(self):
        X = np.random.uniform(-1, 1, (30, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = QuantumClassifier(n_qubits=2, skip_preprocessing=True, seed=42)
        clf.fit(X, y)
        assert clf.score(X, y) >= 0.0
