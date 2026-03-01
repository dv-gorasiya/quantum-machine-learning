"""Quantum classifier with automatic preprocessing and sklearn-compatible API."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from qiskit.circuit.library import zz_feature_map, n_local
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class QuantumClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible quantum classifier with built-in preprocessing.

    Wraps Qiskit Machine Learning's QSVC and VQC behind a familiar fit/predict
    interface. Handles PCA dimensionality reduction, feature scaling, and
    quantum circuit construction automatically.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (and PCA components) to use. Must be <= number of
        features in the dataset.
    method : {"qsvc", "vqc"}
        "qsvc" uses a quantum kernel with SVM. "vqc" uses a variational
        quantum circuit trained with a classical optimizer.
    feature_map_reps : int
        Repetitions of the ZZ feature map circuit.
    entanglement : {"linear", "full", "circular"}
        Entanglement strategy for the feature map.
    optimizer_maxiter : int
        Maximum optimizer iterations (VQC only).
    seed : int or None
        Random seed for reproducibility.
    skip_preprocessing : bool
        If True, skip PCA and scaling. Use when data is already prepared
        (n_features == n_qubits, values in [-1, 1]).

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> y = (y == 0).astype(int)  # binary classification
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> clf = QuantumClassifier(n_qubits=2)
    >>> clf.fit(X_train, y_train)
    >>> print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
    """

    def __init__(
        self,
        n_qubits: int = 2,
        method: str = "qsvc",
        feature_map_reps: int = 2,
        entanglement: str = "linear",
        optimizer_maxiter: int = 100,
        seed: int | None = 42,
        skip_preprocessing: bool = False,
    ):
        self.n_qubits = n_qubits
        self.method = method
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement
        self.optimizer_maxiter = optimizer_maxiter
        self.seed = seed
        self.skip_preprocessing = skip_preprocessing

    def _validate_params(self):
        if self.method not in ("qsvc", "vqc"):
            raise ValueError(f"method must be 'qsvc' or 'vqc', got '{self.method}'")
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")

    def _build_preprocessing(self, n_features: int):
        if self.skip_preprocessing:
            return

        if self.n_qubits > n_features:
            raise ValueError(
                f"n_qubits ({self.n_qubits}) cannot exceed number of features "
                f"({n_features}). Reduce n_qubits or add more features."
            )

        self._scaler = StandardScaler()
        self._pca = PCA(n_components=self.n_qubits)
        self._minmax = MinMaxScaler(feature_range=(-1, 1))

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.skip_preprocessing:
            return X

        if fit:
            X = self._scaler.fit_transform(X)
            X = self._pca.fit_transform(X)
            X = self._minmax.fit_transform(X)
        else:
            X = self._scaler.transform(X)
            X = self._pca.transform(X)
            X = self._minmax.transform(X)
        return X

    def _build_model(self):
        fmap = zz_feature_map(
            feature_dimension=self.n_qubits,
            reps=self.feature_map_reps,
            entanglement=self.entanglement,
        )

        if self.method == "qsvc":
            kernel = FidelityQuantumKernel(feature_map=fmap)
            return QSVC(quantum_kernel=kernel)

        ansatz = n_local(
            fmap.num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            reps=3,
        )
        return VQC(
            feature_map=fmap,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=self.optimizer_maxiter),
        )

    def fit(self, X, y):
        """Fit the quantum classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_params()
        X, y = check_X_y(X, y)

        if self.seed is not None:
            algorithm_globals.random_seed = self.seed

        self._build_preprocessing(X.shape[1])
        X = self._preprocess(X, fit=True)

        self.model_ = self._build_model()
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        X = check_array(X)
        X = self._preprocess(X, fit=False)
        return self.model_.predict(X)

    def score(self, X, y):
        """Return classification accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        accuracy : float
        """
        check_is_fitted(self, "model_")
        X = check_array(X)
        X = self._preprocess(X, fit=False)
        return self.model_.score(X, y)
