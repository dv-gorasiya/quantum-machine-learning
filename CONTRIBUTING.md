# Contributing

Contributions are welcome. This project aims to lower the barrier to getting started with Quantum Machine Learning.

## Getting Started

```bash
git clone https://github.com/dv-gorasiya/quantum-machine-learning.git
cd quantum-machine-learning
pip install -e ".[dev]"
pytest
```

## What to Contribute

- **New dataset loaders** — add preprocessed loaders in `qml_starter/datasets.py`
- **Additional quantum methods** — e.g., Pegasos QSVC, Quantum Neural Network classifiers
- **Benchmarks** — results on different datasets, qubit counts, or feature maps
- **Documentation** — improve docstrings, add tutorials
- **Bug fixes** — especially around edge cases in preprocessing

## Guidelines

1. Open an issue first to discuss non-trivial changes.
2. Keep PRs focused — one feature or fix per PR.
3. Add tests for new functionality.
4. Run `pytest` before submitting.
5. Follow existing code style (type hints, docstrings, sklearn conventions).

## Code Style

- Type hints on all public function signatures.
- Google-style docstrings with `Parameters`, `Returns` sections.
- sklearn-compatible API patterns (`fit`, `predict`, `score`).
