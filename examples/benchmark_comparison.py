"""Compare quantum vs classical classifiers across datasets."""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from qml_starter.benchmark import compare, print_comparison

datasets = {
    "Iris (0 vs 1)": load_iris,
    "Breast Cancer": load_breast_cancer,
}

for name, loader in datasets.items():
    data, target = loader(return_X_y=True)

    # For Iris, make it binary
    if "Iris" in name:
        mask = target < 2
        data, target = data[mask], target[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )

    print(f"\n{'=' * 47}")
    print(f" Dataset: {name}")
    print(f" Samples: {len(data)} | Features: {data.shape[1]}")
    print(f"{'=' * 47}")

    results = compare(X_train, y_train, X_test, y_test, n_qubits=2)
    print_comparison(results)
