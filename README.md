# Amazon ML Challenge 2026 - Entity Resolution

This repository contains the final source code and methodology for the **Amazon ML Challenge 2026**. The goal of this challenge is to perform large-scale **Entity Resolution** (deduplication) across noisy, multilingual, and highly variable business datasets.

## 🏆 Performance
- **Validation F0.5 Score:** `~0.9869`
- **Metric Focus:** The F0.5 metric heavily penalizes false merges (precision is weighted 2x over recall). Our pipeline is meticulously tuned to maximize this precision-heavy objective.

## 🧠 Methodology & Pipeline Overview

The solution uses a highly optimized **Multi-Stage Lexical & Gradient Boosting Pipeline** designed to scale across a universe of 10.3 million entities. 

### 1. Blocking & Candidate Generation (Stage 1)
To reduce the $O(N^2)$ search space, we utilize high-recall blocking techniques:
- **TF-IDF + `sparse_dot_topn`:** Ultra-fast sparse matrix multiplication (C++ accelerated) is used on normalized business names and addresses to fetch the Top-K candidates.
- **Fuzzy String Similarity:** Integration of rapidfuzz metrics (Jaccard, Levenshtein, token set ratios).

### 2. Candidate Classification (Stage 2 & 3)
A multi-stage tree-based ensemble filters the initial candidates:
- **Feature Engineering:** We extract cross-entity features including length absolute differences, character token overlaps, numeric agreement bounding (preventing street number collisions), and cluster-density sibling features.
- **Stage-2 LightGBM:** Two cross-fitted LightGBM models score the raw candidate pairs.
- **Stage-3 Iterative Boosting:** A final LightGBM model re-evaluates the pairs using density/group features recalculated from the Stage-2 probability distributions.

### 3. Expected-F0.5 Thresholding (The E09 Rule)
Instead of a static probability threshold (e.g., `P > 0.5`), we dynamically optimize the F0.5 score on a per-query basis. By sorting a query's candidates by probability and calculating the mathematical expectation of the F0.5 metric, the model dynamically halts candidate selection when the expected score drops, perfectly balancing singletons against multi-match queries.

## 📂 Repository Structure
- `business_entity_resolution/src/` - Core reusable pipeline modules (normalization, features, blocking).
- `entity_resolution_solution/` - Custom diagnostics and missed-match analysis scripts.
- `experiments/` - Ledger and incremental code for iterative experiments (E00 to E10).
  - `final_predict.py` - The final entry point to generate predictions on the test set.
- `output/` - Location for the generated `matching_results.tsv` and `candidate_pairs.tsv`.
- `Documentation_template.md` - In-depth methodology write-up.

## 🚀 How to Reproduce
1. Install dependencies:
   ```bash
   pip install -r business_entity_resolution/requirements.txt
   ```
2. Place the unzipped `dataset/` folder inside the root directory.
3. Run the end-to-end pipeline:
   ```bash
   python experiments/final_predict.py experiments/config_E09.json
   ```

*Developed for the Amazon ML Challenge 2026.*
