<div align="center">
  <h1>🌟 Amazon ML Challenge 2026</h1>
  <h3>Business Entity Resolution & Deduplication</h3>
  <i>An ultra-fast, multi-stage LightGBM pipeline designed to deduplicate highly noisy business records across a 10.3 million entity universe.</i>
</div>

<br/>

## 🏆 Performance

| Metric | Score |
| :--- | :--- |
| **Final Leaderboard F0.5** | **`0.972865`** |
| **Validation F0.5** | `0.9869` |

> **Note on the Metric:** The competition is judged on the **Macro F0.5 Score**, which penalizes false merges (false positives) twice as heavily as missed matches (false negatives). Correctly identifying singletons (entities with zero matches) yields a perfect 1.0, while a single false merge on a singleton drops the score to 0.0. Our pipeline was mathematically optimized around this asymmetric risk.

---

## 📖 Problem Statement

With the rapid expansion of global commerce, e-commerce platforms aggregate millions of business registrations. This creates massive fragmentation: a single business might register multiple times across different channels with variations in their name, language, spelling, or address formatting.

**The Task:** Given a target universe of business records (**Source 1**), identify all true matching counterparts in two auxiliary, highly noisy databases (**Source 2** and **Source 3**). 

**The Challenges:**
1. **Scale:** Over 10 million combined records. Quadratic $O(N^2)$ comparisons are impossible.
2. **Noise:** Severe typographical errors, missing fields, arbitrary abbreviations, and cross-lingual phonetic spellings.
3. **The Singleton Trap:** Over 5% of queries have *no* valid matches. Aggressive retrieval algorithms that force matches onto singletons destroy the precision-heavy F0.5 score.

---

## 🧠 Solution Architecture

To tackle the scale and precision requirements, we engineered a **3-Stage Lexical & Gradient Boosting Pipeline**:

### Stage 1: High-Recall Blocking & Candidate Generation
Comparing every record against 10.3 million targets is computationally impossible. We drastically reduce the search space while maintaining >95% recall:
* **Lexical Indexing:** We construct Word and Character-Ngram TF-IDF matrices for all business names and addresses.
* **C++ Accelerated Sparse Retrieval:** Using `sparse_dot_topn`, we perform blazingly fast sparse matrix multiplications to fetch the Top-K closest candidates.
* **Fuzzy Filtering:** We integrate `rapidfuzz` string metrics (Jaccard, Levenshtein, token set ratios) to prune obvious false positives.

### Stage 2: Pairwise Candidate Classification
The surviving candidate pairs are passed into a highly tuned LightGBM classifier:
* **Feature Engineering:** We extract cross-entity features such as absolute length differences, numeric agreement (ensuring street numbers/pincodes don't collide), and character token overlaps.
* **Cross-Fitted Ensembles:** We train multiple LightGBM models on K-Fold splits of the data to prevent overfitting and average their probabilistic outputs.

### Stage 3: Density-Aware Re-ranking
Because matches aren't independent (if A matches B, and A matches C, B and C should be similar), we feed Stage-2 probabilities back into a final model:
* **Sibling / Cluster Features:** We compute "density" features that measure the confidence of other candidates in the same query pool.
* **Iterative Boosting:** A final LightGBM model processes these contextual group features to correct anomalous predictions.

---

## ✨ Key Innovations

1. **Expected-F0.5 Thresholding (The E09 Rule):**
   Standard pipelines use a static threshold (e.g., `Probability > 0.5`). Because F0.5 is non-linear and heavily penalizes singleton errors, we dynamically optimize the threshold *per query*. We sort candidates by probability and compute the mathematical expectation of the F0.5 metric, halting selection the exact moment the expected score drops.
2. **Numeric Collision Bounding:**
   Treating missing numbers as a conflict destroys recall. We engineered features that resolve `num_agree` and `num_conflict` intelligently, ensuring that missing suite numbers don't unjustly penalize genuine matches.

---

## 📂 Repository Structure

```text
├── business_entity_resolution/
│   ├── src/                 # Core reusable modules (normalization, features, blocking)
│   └── requirements.txt     # Python dependencies
├── entity_resolution_solution/
│   └── src/                 # Diagnostics, missed-match analysis, and research scripts
├── experiments/             # Ledger and incremental pipeline code (E00 to E10)
│   └── final_predict.py     # Final inference script used for submission
├── output/                  # Directory for generated TSVs (matching_results.tsv)
└── Documentation_template.md # Detailed methodology write-up
```

---

## 🚀 Quick Start / Reproduction

**1. Environment Setup**
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r business_entity_resolution/requirements.txt
```

**2. Data Placement**
Ensure the competition `dataset/` folder (containing `train/` and `test/` splits) is placed in the root directory.

**3. Run Final Inference**
To run the full end-to-end pipeline and generate the final `matching_results.tsv`:
```bash
python experiments/final_predict.py experiments/config_E09.json
```

<br/>

<div align="center">
  <i>Developed for the Amazon ML Challenge 2026</i>
</div>
