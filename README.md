# Contextual Inverse Learning (CIL)

Code repository for the paper:

> **Solution Prediction Dominates Parameter Prediction Under Generic Non-Identifiability: Connecting Inverse Optimization and Predict-then-Optimize**  
> Farzin Ahmadi, Fardin Ganjkhanloo  
> *European Journal of Operational Research* (under review)

---

## Overview

This paper shows that when cost parameters are non-identifiable from observed decisions — which is the generic case for LPs with more constraints than variables — the standard two-stage inverse optimization pipeline fails irreducibly. **Contextual Inverse Learning (CIL)** bypasses parameter recovery entirely by learning a direct context-to-decision mapping, achieving up to 97% lower decision suboptimality than two-stage baselines.

---

## Repository structure

```
cil-repo/
├── experiments/
│   ├── reviewer_experiments.py     # Main script: hypothesis classes, shortest path, ISPO+ warm-start variants
│   └── reviewer_experiments_v2.py  # Corrected version: fixed test set, proper SP framing
├── figures/
│   ├── generate_figures.py         # Reproduces all paper figures from hardcoded results
│   └── output/                     # Pre-generated PDF figures
│       ├── fig_main_paper.pdf
│       ├── fig_hypothesis_class.pdf
│       ├── fig_ispo_warmstart.pdf
│       ├── fig_shortest_path.pdf
│       └── ...
├── results/
│   ├── results_main.csv            # Raw results (all methods, all configurations)
│   └── results_final.csv           # Aggregated results used in the paper tables
├── requirements.txt
└── README.md
```

---

## Reproducing the results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

All experiments use **scipy** for LP solving (no commercial solver required).

### 2. Run the experiments

The main experimental results in the paper (27-configuration full factorial design, 50 trials each) are stored in `results/`. To reproduce the reviewer-response experiments (hypothesis classes, shortest-path case study, ISPO+ warm-start analysis):

```bash
python experiments/reviewer_experiments_v2.py
```

This runs the corrected version with:
- A fixed held-out test set across training sizes (Comment 2)
- Shortest-path observation model `y = θ + noise` with a Uniform baseline (Comment 1)
- Four ISPO+ initialization variants (Comment 3)

### 3. Regenerate figures

```bash
python figures/generate_figures.py
```

Outputs PDF and PNG figures to `figures/output/`.

---

## Methods compared

| Method | Description |
|---|---|
| **IO + LS** | Two-stage: NNLS parameter recovery → ridge regression |
| **IO + SPO+** | Two-stage: NNLS recovery → SPO+ loss in Stage 2 |
| **CIL (ISPO+)** | Direct solution prediction with Inverse SPO+ loss |
| **CIL (MSE)** | Direct solution prediction with mean-squared-error loss |

---

## Key results

- CIL (MSE) achieves **97% lower normalized SPO loss** than IO + LS across 27 configurations
- CIL (ISPO+) achieves **72% lower normalized SPO loss** than IO + LS
- Two-stage methods plateau at ~200% suboptimality regardless of training set size or noise level — consistent with the theoretical error floor (Corollary 1)
- Neural network predictors reduce decision error by **46% over linear models** at K = 500

---

## Contact

Farzin Ahmadi — fahmadi@towson.edu  
Fardin Ganjkhanloo — fganjkh1@jhu.edu
