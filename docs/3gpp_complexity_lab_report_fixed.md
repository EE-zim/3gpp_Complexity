# Quantifying Complexity Across 3GPP Releases  

> *“How much harder does it get to engineer a full‑stack implementation as the 3GPP standard evolves?”*  

This notebook answers the question by  

1. embedding every normative sentence of 3GPP Releases 8–18 (plus early Rel‑19) with a **domain‑tuned SBERT** model,  
2. computing five semantic‑complexity metrics for each Release,  
3. collapsing them into a single **Engineering‑Load Index (EFI / EELI)**, and  
4. visualising how the load has grown over fifteen years of standardisation.  




## 1. Data & Pre‑processing  

* **Corpora** – concatenated `*.docx/TS/TR` for the main radio‑access specs of each Release.  
* **Sentence splitting** – spaCy `en_core_web_lg` + manual fixes for ASN.1 blocks.  
* **Embedding** – `SBERT` . Maximum sequence length =512  token; longer sentences are window‑pooled.  

Datasets already reside in two CSVs:  

| file | purpose |
|------|---------|
| `release_metrics.csv` | per‑Release absolute metrics & sentence counts |
| `delta_metrics.csv`   | inter‑Release deltas (CM, ND) |


## 2. Complexity Metrics  


| Symbol | Name | Formula | Captures |
|--------|------|---------|----------|
| SS | *Semantic Spread* | $\operatorname{Tr}\!\bigl(\operatorname{Cov}(\mathbf{s})\bigr)$ | Breadth of semantic space |
| RI | *Redundancy Index* | $1-\bar{\,\cos(\mathbf{s}_i,\mathbf{s}_j)\,}$ | Non‑repetition |
| CE | *Cluster Entropy* | $H\_k = -\sum\_{c=1}^k p_c\log p_c$ (k‑means, $k\approx\sqrt N$) | Topic balance |
| CM | *Change Magnitude* | $1-\cos(\mu_r,\mu_{r+1})$ | Shift of centroid between Releases |
| ND | *Novelty Density* | $\displaystyle\frac1N\sum_i \min_j \|\mathbf{s}_i-\mathbf{s}_j^{(prev)}\|$ | Sentence‑level newness |  

All vectors $\mathbf{s}$ live in the SBERT 768‑D space.  


## 3. Star plot

### Semantic Spread (SS)

### Redundancy Index (RI)

# Novelty Density (ND)



### 1 Composite Metric

$$
\boxed{\; \mathrm{EFI}_r \;=\; S_r \;\times\; D_r \;\times\; \bigl(1 + V_r\bigr) \;}
$$

*EFI = Engineering Footprint Index for release **r** — a single number estimating end-to-end implementation effort.*

$$
\mathrm{EELI}_r \;=\; \log_{10}\!\bigl(\mathrm{EFI}_r\bigr)
$$

*EELI (Engineering Load Index) compresses EFI to a log scale; +1 ≈ 10 × more work.*

---

| Symbol   | Formula                                                                            | Captures                                                                                     | Typical range |
| -------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------- |
| $S\_r$ | \$\log\_{10} N\_r\$                                                                | **Size** — quantity of normative “atoms” (sentences, pages, or words)                        | 3 – 5         |
| \$D\_r\$ | \$\displaystyle \frac{\mathrm{CE}\_r ,\mathrm{SS}\_r}{\mathrm{RI}\_r}\$            | **Diversity** — breadth & balance of topics after deducting redundancy                       | 1 – 12        |
| \$V\_r\$ | \$\displaystyle \alpha,\mathrm{CM}*{r-1!\to r} ;+; \beta,\mathrm{ND}*{r-1!\to r}\$ | **Velocity** — how much genuinely new or disruptive content appears vs. the previous release | 0 – 0.15      |






### Constructing a Composite Index: EFI

We define the **Engineering-Footprint Index (EFI)** for release *r* as:

$$
\text{EFI}_r = S_r \cdot D_r \cdot (1 + V_r)
$$

Where:

* $S_r$: **Scale** – the overall volume of features to implement
* $D_r$: **Diversity** – the degree of heterogeneity across functionalities
* $V_r$: **Velocity** – the relative rate of newly added features

This is a **linear multiplicative** model that combines the three dimensions of engineering complexity.

---

### Exponential Trend:

* In practice, $S_r$ tends to **increase linearly** over time.
* $D_r$ rises slowly at first and then plateaus, but **remains >1**.
* $V_r$ **spikes sharply** before and after each major generational shift (e.g., 4G → 5G).

When combined, these factors produce an **approximately exponential or stepwise escalation** in EFI, which from an engineering perspective explains the **explosion in the testing matrix**.

---

### Logarithmic Visualization:

To aid interpretation, we define the **Engineering Effort Log-Index (EELI)** as the base-10 logarithm of EFI:

$$
\text{EELI}_r = \log_{10}(\text{EFI}_r)
$$

This transformation makes it easier to visualize and compare engineering burdens across different 3GPP releases.



### **3GPP Release Mapping Table**

| Release | Key Features                                  | Corresponding Generation | Release Year       |
| ------- | --------------------------------------------- | ------------------------ | ------------------ |
| Rel-8   | Introduction of **LTE (Long Term Evolution)** | Initial 4G               | 2008               |
| Rel-9   | LTE enhancements (e.g., Dual-layer MIMO)      | 4G                       | 2009               |
| Rel-10  | Introduction of **LTE-Advanced**              | Enhanced 4G              | 2011               |
| Rel-11  | Enhanced Carrier Aggregation                  | 4G                       | 2012               |
| Rel-12  | Support for D2D, Small Cells, etc.            | 4G / Transitional        | 2014               |
| Rel-13  | LTE-Advanced Pro (e.g., NB-IoT)               | 4.5G / Pre-5G            | 2016               |
| Rel-14  | V2X, eMTC, and more                           | 4.5G                     | 2017               |
| Rel-15  | **First 5G NR Standard** (NSA/SA)             | **Initial 5G**           | 2018               |
| Rel-16  | 5G enhancements (e.g., URLLC, V2X)            | 5G                       | 2020               |
| Rel-17  | 5G expansion (e.g., RedCap, NTN)              | Enhanced 5G              | 2022               |
| Rel-18  | **First phase of 5G-Advanced**                | 5.5G                     | 2024      |
| Rel-19  | **Second phase of 5G-Advanced**               | 5.5G / Towards 6G        | Expected 2025–2026 |

---

### Summary:

* **Rel-8 to Rel-14**: Mainly cover **4G LTE and its evolution** (also known as LTE-A, LTE-A Pro).
* **Rel-15 to Rel-17**: Represent various stages of **5G NR (New Radio)**.
* **Rel-18 to Rel-19**: Fall under **5G-Advanced (5.5G)** and serve as a **stepping stone toward 6G**.
* **Releases prior to Rel-6/Rel-7**: Primarily represent **3G (UMTS/HSPA)**.
