# 3GPP Complexity Metrics

This repository contains scripts for computing semantic complexity metrics on 3GPP specification text. The main entry points are:

- `tspec_metrics_2.py` – feature rich pipeline with optional Weights & Biases logging.
- `tspec_metrics_HPC.py` – simplified version suitable for batch processing or HPC usage.

Both scripts share metric helper functions such as `semantic_spread`, `redundancy_index`, and `cluster_entropy`.

## Running the Pipeline

1. Ensure dependencies are installed:
   ```bash
   pip install numpy pandas torch spacy sentence-transformers scikit-learn tqdm
   ```
2. Prepare your specification root directory (`SPEC_ROOT` environment variable) containing `Rel-*` folders.
3. Execute the script:
   ```bash
   python tspec_metrics_2.py --reset-checkpoint
   ```
   or on HPC:
   ```bash
   python tspec_metrics_HPC.py --reset-checkpoint
   ```

Metrics will be written to `release_metrics.csv` and `delta_metrics.csv`.

## Running Tests

Unit tests validate metric computations for small arrays:

```bash
pytest -q
```

`pytest` is optional but recommended for verifying correctness.
