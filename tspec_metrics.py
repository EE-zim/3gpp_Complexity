# !pip install sentence-transformers pdfminer.six spacy tqdm scikit-learn pandas numpy scipy
import os, re, sys
from pathlib import Path; from glob import glob
from typing import List, Dict
import numpy as np, pandas as pd, torch, spacy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from scipy.stats import entropy

# Parameters
spec_root = Path('~/Documents/WorkSpace/TSpec-LLM/3GPP-clean-test').expanduser()  # <-- EDIT ME
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
block_size  = 200_000      # chars per spaCy chunk
batch_size  = 128
sample_redundancy = 1000
sample_novelty    = 2000

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def extract_text(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in ('.txt', '.md'):
        return p.read_text(encoding='utf-8', errors='ignore')
    if suf == '.pdf':
        import pdfminer.high_level as pdfm
        return pdfm.extract_text(p)
    raise ValueError(f'Unsupported {p}')
    
def make_nlp(max_len=5_000_000):
    nlp = spacy.blank('en'); nlp.max_length = max_len; nlp.add_pipe('sentencizer'); return nlp

def block_split(nlp, text: str, blk_sz=200_000):
    s=[]
    for i in range(0,len(text),blk_sz):
        blk=text[i:i+blk_sz]
        if len(blk)>=nlp.max_length: nlp.max_length=len(blk)+1
        s.extend(t.text.strip() for t in nlp(blk).sents if t.text.strip())
    return s

def semantic_spread(X): return float(np.trace(np.cov(X,rowvar=False)))
def redundancy_index(X, k=1000):
    if len(X)>k: X=X[np.random.choice(len(X),k,False)]
    sims=util.cos_sim(X,X).cpu().numpy(); return 1.0-float(sims[np.triu_indices_from(sims,1)].mean())
def cluster_entropy(X):
    labels=KMeans(n_clusters=int(np.sqrt(len(X))),n_init='auto',random_state=0).fit_predict(X)
    p=np.bincount(labels)/len(labels); return float(entropy(p,base=2))
def change_mag(a,b): return 1.0-float(util.cos_sim(a,b))
def novelty_density(Xp,Xn,k=2000):
    if len(Xn)>k: Xn=Xn[np.random.choice(len(Xn),k,False)]
    sims=util.cos_sim(Xn,Xp).cpu().numpy(); return float((1.0-sims.max(1)).mean())
    
    # Discover releases & files
releases=sorted([d.name for d in spec_root.iterdir() if d.is_dir() and d.name.startswith('Rel-')],
                key=lambda x:int(re.findall(r'\d+',x)[0]))
print('Releases:', ', '.join(releases))
rel_files={r:[Path(p) for p in glob(str((spec_root/r)/'**/*'),recursive=True) if p.lower().endswith(('.txt','.md','.pdf'))] for r in releases}
for r,fl in rel_files.items(): print(r,len(fl),'files')

# Gather all sentences
nlp=make_nlp()
all_sents=[]; rel_slice={}
for r in releases:
    start=len(all_sents)
    for f in tqdm(rel_files[r],desc=f'{r} files',unit='file',leave=False):
        try: txt=extract_text(f)
        except Exception as e: print('[warn]',f,e); continue
        all_sents.extend(block_split(nlp,txt,block_size))
    rel_slice[r]=slice(start,len(all_sents))
print('Total sentences:',len(all_sents))

# Encode once
model=SentenceTransformer(model_name,device=device)
emb=model.encode(all_sents,batch_size=batch_size,device=device,normalize_embeddings=True,show_progress_bar=True)
X_all=np.asarray(emb)

# Metrics per release
metrics=[]; mus={}; mats={}
for r in releases:
    s=rel_slice[r]; X=X_all[s]; 
    if X.size==0: print('[skip]',r); continue
    mus[r]=X.mean(0); mats[r]=X
    metrics.append({'release':r,'sentences':len(X),
                    'semantic_spread':semantic_spread(X),
                    'redundancy_index':redundancy_index(X,sample_redundancy),
                    'cluster_entropy':cluster_entropy(X)})
df_rel=pd.DataFrame(metrics); df_rel.to_csv('release_metrics.csv',index=False); df_rel

# Delta metrics
delta=[]
for a,b in zip(releases[:-1],releases[1:]):
    if a in mus and b in mus:
        delta.append({'from':a,'to':b,
                      'change_magnitude':change_mag(mus[a],mus[b]),
                      'novelty_density':novelty_density(mats[a],mats[b],sample_novelty)})
df_delta=pd.DataFrame(delta); df_delta.to_csv('delta_metrics.csv',index=False); df_delta