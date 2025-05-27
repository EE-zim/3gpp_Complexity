# 3GPP Complexity Metrics

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/EE-zim/3gpp_Complexity)

This repository contains scripts for computing semantic complexity metrics on 3GPP specification text. The main entry points are:

The repository is organised as follows:

- `src/` – Python modules containing the metric implementations and pipelines
  (`metric_utils.py`, `tspec_metrics_2.py`, `tspec_metrics_HPC.py`).
- `notebooks/` – Jupyter notebooks used for exploration and reporting.
- `docs/` – Markdown exports of notebooks and additional documentation.
- `data/` – CSV and NumPy files storing computed metrics.

Metric helper functions such as `semantic_spread`, `redundancy_index`, and
`cluster_entropy` live in `src/metric_utils.py` and are used by the pipeline scripts.

## Project Overview

`3gpp_complexity_lab_report_fixed.ipynb` demonstrates how we quantify complexity across 3GPP releases. Every normative sentence from Release 8 onward is embedded with a domain-tuned SBERT model. We compute five metrics—Semantic Spread (SS), Redundancy Index (RI), Cluster Entropy (CE), Change Magnitude (CM), and Novelty Density (ND)—and combine them into an Engineering Footprint Index (EFI) and its log form, the Engineering Load Index (EELI). A Markdown export of the notebook is available under `docs/3gpp_complexity_lab_report_fixed.md` for convenient browsing outside of Jupyter.
The computed index reveals a steady rise in complexity, from about 6.72 in Release 8 to over 7.19 by Release 17 (on the log-scaled EELI). Values are stored in `data/release_metrics.csv` and `data/delta_metrics.csv`. The notebook concludes with a mapping of releases to their associated 4G and 5G milestones.


## Running the Pipeline

1. Ensure dependencies are installed:
   ```bash
   pip install numpy pandas torch spacy sentence-transformers scikit-learn tqdm
   ```
2. Prepare your specification root directory (`SPEC_ROOT` environment variable) containing `Rel-*` folders.
3. Execute the script:
   ```bash
   python src/tspec_metrics_2.py --reset-checkpoint
   ```
   or on HPC:
   ```bash
   python src/tspec_metrics_HPC.py --reset-checkpoint
   ```

Metrics will be written to `data/release_metrics.csv` and `data/delta_metrics.csv`.

## Running Tests

Unit tests validate metric computations for small arrays:

```bash
pytest -q
```

`pytest` is optional but recommended for verifying correctness.
