#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import re
import pickle
import argparse
import multiprocessing
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
import spacy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from scipy.stats import entropy


def semantic_spread(X):
    return float(np.trace(np.cov(X, rowvar=False)))


def redundancy_index(X, k=1000):
    if len(X) > k:
        X = X[np.random.choice(len(X), k, replace=False)]
    sims = util.cos_sim(X, X).cpu().numpy()
    return 1.0 - float(sims[np.triu_indices_from(sims, 1)].mean())


def cluster_entropy(X, sample=5000):
    if len(X) > sample:
        X = X[np.random.choice(len(X), sample, replace=False)]
    labels = KMeans(n_clusters=int(np.sqrt(len(X))),
                    n_init='auto', random_state=0).fit_predict(X)
    p = np.bincount(labels) / len(labels)
    return float(entropy(p, base=2))


def change_mag(a, b):
    return 1.0 - float(util.cos_sim(a, b))


def novelty_density(Xp, Xn, k=2000):
    if len(Xn) > k:
        Xn = Xn[np.random.choice(len(Xn), k, replace=False)]
    sims = util.cos_sim(Xn, Xp).cpu().numpy()
    return float((1.0 - sims.max(1)).mean())


def parse_args():
    parser = argparse.ArgumentParser(description="Compute TSpec-LLM release metrics with checkpointing and parallelism.")
    parser.add_argument('--reset-checkpoint', action='store_true',
                        help='Delete old checkpoint and recompute sentence splits')
    parser.add_argument('--checkpoint-file', default='checkpoint.pkl',
                        help='Checkpoint filename (default: checkpoint.pkl)')
    return parser.parse_args()


def get_sentences(spec_root: Path, block_size: int, cpus: int,
                  checkpoint_file: str, reset_checkpoint: bool):
    cp = Path(checkpoint_file)
    if cp.exists() and not reset_checkpoint:
        print(f'Loading sentences from checkpoint: {checkpoint_file}')
        with cp.open('rb') as f:
            return pickle.load(f)

    if cp.exists() and reset_checkpoint:
        cp.unlink()
        print(f'Removed old checkpoint: {checkpoint_file}')

    print('Starting sentence splitting phase...')
    nlp = spacy.blank('en')
    nlp.max_length = max(block_size, 2_000_000)
    nlp.add_pipe('sentencizer')

    # discover releases
    releases = sorted(
        [d.name for d in spec_root.iterdir() if d.is_dir() and d.name.startswith('Rel-')],
        key=lambda x: int(re.findall(r"\d+", x)[0])
    )
    print('Releases:', ', '.join(releases))

    all_sents = []
    rel_slice = {}
    for r in releases:
        start = len(all_sents)
        blocks = []
        for f in (spec_root / r).rglob('*'):
            if f.suffix.lower() in ('.txt', '.md', '.pdf'):
                try:
                    text = (f.read_text(encoding='utf-8', errors='ignore')
                            if f.suffix.lower() in ('.txt', '.md')
                            else __import__('pdfminer.high_level').extract_text(str(f)))
                except Exception as e:
                    print(f'[warn] could not read {f}: {e}')
                    continue
                for i in range(0, len(text), block_size):
                    blocks.append(text[i:i+block_size])

        # parallel sentence splitting
        for doc in tqdm(nlp.pipe(blocks,
                                 batch_size=64,
                                 n_process=cpus),
                        desc=f'Splitting {r}', unit='chunk'):
            for sent in doc.sents:
                s = sent.text.strip()
                if s:
                    all_sents.append(s)
        rel_slice[r] = slice(start, len(all_sents))

    print(f'Collected {len(all_sents):,} sentences, saving checkpoint to {checkpoint_file}')
    with cp.open('wb') as f:
        pickle.dump((all_sents, rel_slice), f)

    return all_sents, rel_slice


def compute_metrics(all_sents: list, rel_slice: dict,
                    model_name: str, batch_size: int, device: str):
    print('Encoding sentences and computing metrics...')
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        all_sents,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    X_all = np.asarray(emb)

    metrics = []
    mus = {}
    mats = {}

    for r, sl in rel_slice.items():
        X = X_all[sl]
        if X.size == 0:
            print(f'[skip] {r}')
            continue
        mus[r] = X.mean(0)
        mats[r] = X
        metrics.append({
            'release': r,
            'sentences': len(X),
            'semantic_spread': semantic_spread(X),
            'redundancy_index': redundancy_index(X),
            'cluster_entropy': cluster_entropy(X)
        })

    df_rel = pd.DataFrame(metrics)
    df_rel.to_csv('release_metrics.csv', index=False)
    print('Saved release_metrics.csv')

    delta = []
    for a, b in zip(df_rel['release'][:-1], df_rel['release'][1:]):
        delta.append({
            'from': a,
            'to': b,
            'change_magnitude': change_mag(mus[a], mus[b]),
            'novelty_density': novelty_density(mats[a], mats[b])
        })

    df_delta = pd.DataFrame(delta)
    df_delta.to_csv('delta_metrics.csv', index=False)
    print('Saved delta_metrics.csv')


def main():
    args = parse_args()
    SPEC_ROOT = Path(os.environ.get('SPEC_ROOT', '~/Documents/WorkSpace/TSpec-LLM/3GPP-clean')).expanduser()
    BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', 200000))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 256))
    CPUS = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

    print(f'Device: {DEVICE}')
    print(f'SPEC_ROOT: {SPEC_ROOT}')
    print(f'BLOCK_SIZE: {BLOCK_SIZE}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'CPUS: {CPUS}')

    multiprocessing.freeze_support()

    all_sents, rel_slice = get_sentences(
        SPEC_ROOT, BLOCK_SIZE, CPUS,
        args.checkpoint_file, args.reset_checkpoint
    )
    compute_metrics(all_sents, rel_slice, MODEL_NAME, BATCH_SIZE, DEVICE)


if __name__ == '__main__':
    main()


# python tspec_metrics_HPC.py --reset-checkpoint
# python tspec_metrics_HPC.py
# #或者第一个分句后：
# python tspec_metrics_HPC.py --reset-checkpoint

