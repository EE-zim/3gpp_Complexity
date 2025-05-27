# 3GPP Complexity Metrics

This repository contains scripts for computing semantic complexity metrics on 3GPP specification text. The main entry point is:

- `tspec_metrics.py` – unified pipeline with optional Weights & Biases logging.
Metric helper functions such as `semantic_spread`, `redundancy_index`, and
`cluster_entropy` are defined in `metric_utils.py`.

> **Note**
> The previous `tspec_metrics_2.py` and `tspec_metrics_HPC.py` have been merged
> into this single script.

## Project Overview

`3gpp_complexity_lab_report_fixed.ipynb` demonstrates how we quantify complexity across 3GPP releases. Every normative sentence from Release 8 onward is embedded with a domain-tuned SBERT model. We compute five metrics—Semantic Spread (SS), Redundancy Index (RI), Cluster Entropy (CE), Change Magnitude (CM), and Novelty Density (ND)—and combine them into an Engineering Footprint Index (EFI) and its log form, the Engineering Load Index (EELI).
The computed index reveals a steady rise in complexity, from about 6.72 in Release 8 to over 7.19 by Release 17 (on the log-scaled EELI). Values are stored in `release_metrics.csv` and `delta_metrics.csv`. The notebook concludes with a mapping of releases to their associated 4G and 5G milestones.


## Running the Pipeline

1. Ensure dependencies are installed:
   ```bash
   pip install numpy pandas torch spacy sentence-transformers scikit-learn tqdm
   ```
2. Prepare your specification root directory (`SPEC_ROOT` environment variable) containing `Rel-*` folders.
3. Execute the script:
   ```bash
   python tspec_metrics.py --reset-checkpoint
   ```
   The same command works on HPC clusters as well.

Metrics will be written to `release_metrics.csv` and `delta_metrics.csv`.

## Running Tests

Unit tests validate metric computations for small arrays:

```bash
pytest -q
```

`pytest` is optional but recommended for verifying correctness.
