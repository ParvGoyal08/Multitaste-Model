# 📊 Multi-Taste Predictor: EDA Report & Actionable Items

**Dataset**: filtered_db.csv (10,374 molecules)  
**Generated**: March 2026  
**Embeddings Analyzed**: Mol2Vec, RDKit, Morgan FP, MACCS Keys, ChemBERTa

---

## 1. Dataset Overview

### 1.1 Class Distribution (Severe Imbalance)
| Taste Class | Count | Percentage |
|-------------|-------|------------|
| **Sweet** | 5,483 | 52.9% |
| **Undefined** | 2,010 | 19.4% |
| **Sour** | 1,489 | 14.4% |
| **Bitter** | 1,362 | 13.1% |
| **Umami** | 30 | **0.3%** ⚠️ |

### ⚠️ Critical Issue: Extreme Umami Imbalance
- Umami has only **30 samples** (0.3%) - severe underrepresentation
- Model may fail to learn Umami patterns effectively

---

## 2. Embedding Quality Summary

| Embedding | Dimensions | Type | Missing | Zero-Var | Status |
|-----------|-----------|------|---------|----------|--------|
| Mol2Vec | 300 | Continuous | 0 | 0 | ✅ Clean |
| RDKit | 212* | Continuous | 0 | 0 | ✅ Cleaned |
| Morgan FP | 2,048 | Binary | 0 | 0 | ✅ Clean |
| MACCS Keys | 167 | Binary | 0 | 6 | ⚠️ Has zero-var |
| ChemBERTa | 768 | Continuous | 0 | 0 | ✅ Clean |

*RDKit originally had 217 features; 5 removed during cleaning (Ipc extreme values, 4 zero-variance)

### 2.1 RDKit Cleaning Applied
- **Removed columns**: `Ipc` (extreme values >10¹⁰), `SMR_VSA8`, `SlogP_VSA9`, `fr_diazo`, `fr_prisulfonamd` (zero variance)
- **Missing values**: 1,808 values imputed with median
- **Final range**: [-104.04, 77,819.94]

---

## 3. Feature-Taste Correlations (Key Findings)

### 3.1 Best Embedding Per Taste Class
| Taste | Best Embedding | Max Correlation | Top Features |
|-------|---------------|-----------------|--------------|
| **Sweet** | Morgan FP | **0.626** | fp_bit_1921, fp_bit_1487 |
| **Bitter** | RDKit | 0.384 | SMR_VSA4, PEOE_VSA7 |
| **Umami** | Morgan FP | 0.410 | fp_bit_1798, fp_bit_633 |
| **Sour** | RDKit | **0.563** | fr_COO, fr_COO2 |
| **Undefined** | MACCS | 0.484 | maccs_146, maccs_140 |

### 3.2 Interpretable Chemical Insights

#### Sweet Molecules Correlate With:
- **Hydroxyl groups** (H-bond acceptors): `NumHAcceptors` (r=0.59), `NHOHCount` (r=0.59)
- **Polar surface area**: `SlogP_VSA2` (r=0.60), `PEOE_VSA1` (r=0.59)
- **Specific MACCS keys**: `maccs_72` (r=0.57), `maccs_54` (r=0.55)

#### Sour Molecules Correlate With:
- **Carboxylic acid groups**: `fr_COO` (r=0.56), `fr_COO2` (r=0.55), `fr_Al_COO` (r=0.47)
- **Negative correlation with sweet indicators**

#### Bitter Molecules Correlate With:
- **Aromatic/lipophilic features**: `SMR_VSA4` (r=0.38), `NumAliphaticCarbocycles` (r=0.37)
- **Nitrogen-containing groups**: aromatic ring systems

#### Umami (Limited Data):
- **Amino acid-like features**: `fr_HOCCN` (r=0.18), `fr_alkyl_carbamate` (r=0.18)
- Correlations weak due to tiny sample size (n=30)

---

## 4. Dimensionality Analysis (PCA)

| Embedding | PCs for 90% Variance | PC1 Explains | Reducibility |
|-----------|---------------------|--------------|--------------|
| Mol2Vec | 38 | 26.0% | **High** - can reduce to ~40 |
| RDKit | 50+ | 22.5% | Low - features independent |
| Morgan FP | 50+ | 1.4% | Very Low - sparse binary |
| MACCS | 50+ | 16.5% | Low |
| ChemBERTa | 50+ | 14.4% | Low |

### Key Insight:
- **Mol2Vec** is most compressible (38 PCs capture 90% variance)
- **Morgan FP** has distributed information - each bit captures unique info

---

## 5. Binary Fingerprint Analysis

### 5.1 Sparsity
| Fingerprint | Avg Bits ON | Sparsity | Zero-Var Bits |
|-------------|-------------|----------|---------------|
| Morgan FP | 33.2 / 2048 | **98.4%** | 0 |
| MACCS Keys | 34.1 / 167 | 79.6% | 6 |

### 5.2 Actionable:
- Morgan FP extremely sparse → consider removing bits with <1% frequency (1,539 bits)
- MACCS has 6 zero-variance bits → remove before modeling

---

## 6. t-SNE Visualization Insights

From the t-SNE plots:
- **Sweet** molecules form distinct clusters in Mol2Vec and Morgan FP spaces
- **Sour** molecules show clear separation in RDKit space
- **Bitter** molecules overlap significantly with other classes
- **Umami** too few samples to form meaningful clusters
- **Undefined** scattered throughout - may be diverse chemical space

---

## 7. 🎯 ACTIONABLE RECOMMENDATIONS

### 7.1 Data-Level Actions
| Priority | Action | Rationale |
|----------|--------|-----------|
| 🔴 HIGH | Address Umami class imbalance | Only 30 samples - consider SMOTE, data augmentation, or treating as separate binary task |
| 🔴 HIGH | Remove 6 zero-variance MACCS bits | Provide no information |
| 🟡 MED | Remove rare Morgan FP bits (<1%) | 1,539 bits rarely set - reduce from 2048 to ~500 |
| 🟡 MED | Consider excluding Undefined class for initial model | May add noise; treat separately |

### 7.2 Model Selection Strategy

#### For Traditional ML (Random Forest, XGBoost):
- **Primary**: Morgan FP (high correlation with Sweet/Umami)
- **Secondary**: MACCS Keys (interpretable, good for Undefined)
- **Preprocessing**: Binary features, no scaling needed

#### For Neural Networks:
- **Primary**: ChemBERTa or Mol2Vec (dense, continuous)
- **Preprocessing**: StandardScaler normalization
- **Architecture**: Consider attention mechanisms for multi-label

#### For Interpretable Models:
- **Use RDKit descriptors** - chemically meaningful features
- Top features directly explain molecular properties

### 7.3 Multi-Label Strategy
```
Recommended Approach:
1. Train 5 binary classifiers (one per taste)
2. Use embedding best suited for each:
   - Sweet: Morgan FP
   - Sour: RDKit
   - Bitter: RDKit or MACCS
   - Umami: Morgan FP (but oversample!)
   - Undefined: MACCS
3. Combine predictions for final multi-label output
```

### 7.4 Embedding Combination (Ensemble)
Consider concatenating:
- **Mol2Vec (300)** + **MACCS (167)** = 467 features
- Provides both learned representations and interpretable structural keys

### 7.5 Preprocessing Pipeline
```python
# Recommended preprocessing
1. RDKit: Already cleaned, ready to use
2. Morgan FP: Remove bits with frequency < 0.01
3. MACCS: Remove 6 zero-variance bits
4. Mol2Vec/ChemBERTa: StandardScaler
5. Handle class imbalance:
   - SMOTE for Umami
   - Class weights in loss function
```

---

## 8. Files Ready for Modeling

| File | Location | Dimensions | Use Case |
|------|----------|------------|----------|
| mol2vec.csv | Embeddings/ | 10,374 × 305 | Neural networks |
| rdkit_descriptors.csv | Embeddings/ | 10,374 × 217 | Interpretable ML |
| morgan_fps.csv | Embeddings/ | 10,374 × 2,053 | Tree models |
| chemtastes_maccs.csv | Embeddings/ | 10,374 × 172 | Tree models |
| chemtastes_chemberta.csv | Embeddings/ | 10,374 × 773 | Deep learning |

---

## 9. Next Steps Checklist

- [ ] Remove zero-variance MACCS bits (6 bits)
- [ ] Optionally reduce Morgan FP dimensionality
- [ ] Implement class weighting or SMOTE for Umami
- [ ] Create train/val/test splits (stratified by taste)
- [ ] Baseline model: Random Forest with Morgan FP
- [ ] Advanced model: Multi-head neural network with ChemBERTa
- [ ] Evaluate per-class metrics (especially Umami recall)

---

*Report generated from eda3.ipynb analysis*
