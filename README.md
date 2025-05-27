# 3GPP Complexity Metrics




## Running the Pipeline

1. Ensure dependencies are installed:
   ```bash
   pip install numpy pandas torch spacy sentence-transformers scikit-learn tqdm
   ```
2. Prepare your specification root directory (`SPEC_ROOT` environment variable) containing `Rel-*` folders.
3. Execute the script:
   ```bash
   ```
   The same command works on HPC clusters as well.

Metrics will be written to `data/release_metrics.csv` and `data/delta_metrics.csv`.

## Running Tests

Unit tests validate metric computations for small arrays:

```bash
pytest -q
```

`pytest` is optional but recommended for verifying correctness.
