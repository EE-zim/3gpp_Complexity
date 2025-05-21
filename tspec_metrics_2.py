#!/usr/bin/env python3
# coding: utf-8
"""
TSpec-LLM release-metrics pipeline  (rev-2025-05-16)

* sentence-split & slice checkpoint      → --checkpoint-file
* sentence embedding (Sentence-BERT)     → --embeds-file
* Weights & Biases (optional)            → --wandb-project
* system monitor to W&B                  → --log-sys  --sys-interval
"""

import os, re, sys, pickle, argparse, datetime as _dt, multiprocessing, threading, time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch, psutil
import spacy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

from tspec_metrics_HPC import (
    semantic_spread,
    redundancy_index,
    cluster_entropy,
    change_mag,
    novelty_density,
)

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

try:
    import wandb
except ImportError:           # make wandb optional
    wandb = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute TSpec-LLM release metrics with checkpointing & W&B logging")
    # checkpoints
    p.add_argument('--reset-checkpoint', action='store_true',
                   help='Delete old sentence-split checkpoint and recompute')
    p.add_argument('--checkpoint-file', default='checkpoint.pkl',
                   help='Sentence-split checkpoint (default: checkpoint.pkl)')
    p.add_argument('--embeds-file', default='embeddings.npz',
                   help='Sentence embeddings checkpoint (.npz)')
    # W&B
    p.add_argument('--wandb-project', default=None, help='Weights & Biases project (disable if omitted)')
    p.add_argument('--wandb-run', default=None, help='Explicit W&B run name (default: auto-timestamp)')
    # system monitor
    p.add_argument('--log-sys', action='store_true', help='log cpu/ram/gpu to W&B')
    p.add_argument('--sys-interval', type=int, default=30, help='system monitor interval (s)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sentence extraction & checkpointing
# ---------------------------------------------------------------------------

def get_sentences(spec_root: Path, block_size: int, cpus: int,
                  checkpoint_file: str, reset_checkpoint: bool
                  ) -> Tuple[List[str], Dict[str, slice]]:
    """Return (all_sentences, release_slices) from checkpoint or recomputation."""
    cp = Path(checkpoint_file)
    if cp.exists() and not reset_checkpoint:
        print(f'[✓] Loading sentences from {cp}')
        with cp.open('rb') as f:
            return pickle.load(f)

    if cp.exists():
        cp.unlink()
        print(f'[i] Removed old checkpoint: {cp}')

    print('[i] Starting sentence-splitting …')
    nlp = spacy.blank('en')
    nlp.max_length = max(block_size, 2_000_000)
    nlp.add_pipe('sentencizer')

    releases = sorted([d.name for d in spec_root.iterdir()
                       if d.is_dir() and d.name.startswith('Rel-')],
                      key=lambda x: int(re.findall(r'\d+', x)[0]))
    print('   Releases:', ', '.join(releases))

    all_sents: List[str] = []
    rel_slice: Dict[str, slice] = {}

    for r in releases:
        start = len(all_sents)
        blocks: List[str] = []

        files = list((spec_root / r).rglob('*'))
        print(f'   [{r}] scanning {len(files)} files')
        for f in files:
            if f.suffix.lower() not in ('.txt', '.md', '.pdf'):
                continue
            try:
                text = (f.read_text(encoding='utf-8', errors='ignore')
                        if f.suffix.lower() in ('.txt', '.md')
                        else __import__('pdfminer.high_level').extract_text(str(f)))
            except Exception as e:
                print(f'[warn] could not read {f}: {e}')
                continue
            blocks.extend(text[i:i + block_size] for i in range(0, len(text), block_size))

        for doc in tqdm(nlp.pipe(blocks, batch_size=64, n_process=cpus),
                        desc=f'Splitting {r}', unit='chunk'):
            all_sents.extend(s.text.strip() for s in doc.sents if s.text.strip())

        rel_slice[r] = slice(start, len(all_sents))
        print(f'   [{r}] → {len(all_sents) - start:,} sentences')

    print(f'[✓] Collected {len(all_sents):,} sentences; saving checkpoint → {cp}')
    with cp.open('wb') as f:
        pickle.dump((all_sents, rel_slice), f)
    return all_sents, rel_slice


# ---------------------------------------------------------------------------
# Sentence embedding with checkpoint
# ---------------------------------------------------------------------------

def get_embeddings(all_sents: List[str], model_name: str, batch_size: int,
                   device: str, embeds_file: str) -> np.ndarray:
    ep = Path(embeds_file)
    if ep.exists():
        print(f'[✓] Loading embeddings from {ep}')
        with np.load(ep) as npz:
            return npz['X_all']
    print('[i] Encoding sentences …')
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(all_sents, batch_size=batch_size, device=device,
                       normalize_embeddings=True, show_progress_bar=True)
    X_all = np.asarray(emb, dtype=np.float32)
    np.savez_compressed(ep, X_all=X_all)
    print(f'[✓] Saved embeddings → {ep}')
    return X_all


# ---------------------------------------------------------------------------
# System monitor (background thread)
# ---------------------------------------------------------------------------

def start_sys_monitor(wb_run, interval: int):
    if wb_run is None:
        return

    gpu = pynvml.nvmlDeviceGetHandleByIndex(0) if pynvml else None

    def loop():
        while True:
            payload = {
                'sys/cpu': psutil.cpu_percent(),
                'sys/ram': psutil.virtual_memory().percent,
            }
            if gpu:
                util_ = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                mem_ = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                payload.update({'sys/gpu': util_.gpu, 'sys/vram': mem_.used / mem_.total * 100})
            wb_run.log(payload)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f'[✓] System monitor started (interval={interval}s)')


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_metrics(X_all: np.ndarray, rel_slice: Dict[str, slice],
                    wb_run: Optional["wandb.sdk.wandb_run.Run"]) -> None:
    print('[i] Computing metrics …')

    # metric helpers imported from tspec_metrics_HPC

    # per-release ----------------------------------------------------------
    metrics, mus, pools = [], {}, {}
    for r, sl in tqdm(rel_slice.items(), desc='Per-release metrics'):
        X = X_all[sl]
        if X.size == 0:
            print(f'[skip] {r} empty')
            continue

        print(f'   [{r}] sentences={len(X):,}')
        mus[r] = X.mean(0)
        pools[r] = X[np.random.choice(len(X), min(len(X), 2000), replace=False)]
        m = {
            'release': r,
            'sentences': len(X),
            'semantic_spread': semantic_spread(X),
            'redundancy_index': redundancy_index(X),
            'cluster_entropy': cluster_entropy(X),
        }
        metrics.append(m)

        if wb_run:
            step = int(re.findall(r'\d+', r)[0])
            wb_run.log({f'release/{k}': v for k, v in m.items() if k != 'release'}, step=step)

    df_rel = pd.DataFrame(metrics)
    df_rel.to_csv('release_metrics.csv', index=False)
    print('[✓] Saved release_metrics.csv')

    # delta ---------------------------------------------------------------
    delta_rows = []
    rel_list = df_rel['release'].tolist()
    for a, b in tqdm(list(zip(rel_list[:-1], rel_list[1:])), desc='Delta metrics'):
        dr = {
            'from': a,
            'to': b,
            'change_magnitude': change_mag(mus[a], mus[b]),
            'novelty_density': novelty_density(pools[a], pools[b]),
        }
        delta_rows.append(dr)
        if wb_run:
            step = int(re.findall(r'\d+', b)[0])
            wb_run.log({f'delta/{k}': v for k, v in dr.items() if k not in ('from', 'to')}, step=step)

    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv('delta_metrics.csv', index=False)
    print('[✓] Saved delta_metrics.csv')

    # upload --------------------------------------------------------------
    if wb_run:
        art = wandb.Artifact('tspec_metrics', type='dataset',
                             description='CSV metrics for TSpec-LLM releases')
        art.add_file('release_metrics.csv')
        art.add_file('delta_metrics.csv')
        wb_run.log_artifact(art)
        print('[✓] Uploaded CSV artifacts to W&B')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # env hyper-params -----------------------------------------------------
    SPEC_ROOT = Path(os.environ.get('SPEC_ROOT', '~/Documents/WorkSpace/TSpec-LLM/3GPP-clean')).expanduser()
    BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', 200_000))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 256))
    CPUS = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

    print(f'>>> Device={DEVICE}  BLOCK_SIZE={BLOCK_SIZE}  BATCH_SIZE={BATCH_SIZE}  CPUS={CPUS}')
    print(f'>>> SPEC_ROOT={SPEC_ROOT}')

    multiprocessing.freeze_support()

    # W&B -----------------------------------------------------------------
    wb_run = None
    if args.wandb_project:
        if wandb is None:
            raise RuntimeError('wandb-project specified but wandb package not installed. `pip install wandb`')
        run_name = args.wandb_run or f"tspec-metrics-{_dt.datetime.now():%Y%m%d-%H%M%S}"
        wb_run = wandb.init(project=args.wandb_project, name=run_name,
                            config=dict(SPEC_ROOT=str(SPEC_ROOT), BLOCK_SIZE=BLOCK_SIZE,
                                        BATCH_SIZE=BATCH_SIZE, CPUS=CPUS,
                                        DEVICE=DEVICE, MODEL_NAME=MODEL_NAME))
        if args.log_sys:
            start_sys_monitor(wb_run, args.sys_interval)

    # pipeline ------------------------------------------------------------
    all_sents, rel_slice = get_sentences(SPEC_ROOT, BLOCK_SIZE, CPUS,
                                         args.checkpoint_file, args.reset_checkpoint)
    X_all = get_embeddings(all_sents, MODEL_NAME, BATCH_SIZE, DEVICE, args.embeds_file)
    compute_metrics(X_all, rel_slice, wb_run)

    if wb_run:
        wb_run.finish()


if __name__ == '__main__':
    main()
