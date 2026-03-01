"""Visualization utilities for quantum ML experiments."""

from __future__ import annotations

import numpy as np


def plot_decision_regions(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    resolution: float = 0.05,
    title: str = "Decision Regions",
) -> None:
    """Plot 2D decision regions for a fitted classifier.

    Only works when the classifier's effective feature space is 2D
    (i.e., n_qubits=2 or the data has 2 features).

    Parameters
    ----------
    clf : fitted classifier with a .predict() method
    X : array of shape (n_samples, 2) — the 2D projected data
    y : array of shape (n_samples,) — labels
    resolution : mesh grid step size
    title : plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_benchmark(results) -> None:
    """Bar chart comparing classifier accuracies from benchmark results.

    Parameters
    ----------
    results : list of BenchmarkResult from qml_starter.benchmark.compare()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    names = [r.name for r in results]
    accuracies = [r.accuracy for r in results]
    colors = ["#6929C4" if "Quantum" in n else "#1192E8" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, accuracies, color=colors, height=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title("Quantum vs Classical Classifier Comparison")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()
