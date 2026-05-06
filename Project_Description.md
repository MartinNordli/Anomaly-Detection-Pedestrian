# Project Proposal: RPCA-Enhanced Anomaly Detection on the UCSD Pedestrian Dataset

## 1. Background and Motivation

Video anomaly detection is the task of identifying frames or regions in surveillance footage that deviate from learned patterns of normal activity. The UCSD Pedestrian Anomaly Dataset (Mahadevan et al., 2010) is one of the most widely used benchmarks in this field, consisting of two subsets — **Ped1** (34 training and 36 test videos, 158×238 resolution) and **Ped2** (16 training and 12 test videos, 240×360 resolution) — captured by stationary cameras overlooking pedestrian walkways. In normal frames, only pedestrians appear; anomalies consist of non-pedestrian entities (cyclists, skateboarders, small carts, vehicles) or unusual pedestrian motion patterns. Frame-level and pixel-level ground truth annotations are provided for the test set.

Anomaly detection in this setting is fundamentally a problem of separating *rare, transient deviations* from a *persistent, structured background*. This decomposition is precisely the mathematical structure that Robust Principal Component Analysis (RPCA) is designed to recover: given an observation matrix **X**, RPCA solves

$$\min_{L,S} \; \|L\|_* + \lambda \|S\|_1 \quad \text{subject to} \quad X = L + S$$

producing a low-rank component **L** (the slowly-varying background) and a sparse component **S** (transient deviations). The UCSD dataset is particularly well-suited to this decomposition: with a fixed camera and pedestrian crowds that statistically average out over time, the empty walkway is genuinely a low-rank signal, while moving objects — both normal pedestrians and anomalous entities — are sparse in the pixel domain.

Despite this natural alignment, modern deep-learning approaches to video anomaly detection typically operate on raw pixel data, requiring the network to *implicitly* learn background-foreground separation as a byproduct of the main task. We hypothesize that explicitly performing this decomposition as a preprocessing step — using a classical, parameter-free, theoretically-grounded method — can improve downstream classification performance, particularly given the small training set sizes characteristic of the UCSD benchmark.

## 2. Research Questions

This project investigates the following questions:

**RQ1 (Primary):** Does RPCA-based foreground extraction, applied as a preprocessing step, improve the frame-level AUC of a CNN classifier for anomaly detection on UCSD Ped1 and Ped2, compared to training the same architecture on raw video frames?

**RQ2:** Which RPCA component carries the most discriminative signal for anomaly classification — the sparse foreground **S**, the low-rank background **L**, or their concatenation `[L, S]`?

**RQ3:** How do the gains from RPCA preprocessing differ between Ped1 (lower resolution, denser crowds, more challenging) and Ped2 (higher resolution, sparser scenes)? This comparison probes whether RPCA helps more when the underlying signal-to-noise ratio is poorer.

**RQ4 (Secondary, time permitting):** Can SVD-based low-rank approximation of the trained CNN's weight matrices reduce model size with minimal AUC degradation, providing a unified low-rank perspective spanning data preprocessing, classical decomposition theory, and modern model compression?

## 3. Methodology

### 3.1 Data Pipeline

We use the UCSD Pedestrian Anomaly Dataset as distributed on Kaggle (`orvile/ucsd-anomaly-dataset`). For each video clip, frames are loaded at native resolution and converted to grayscale (UCSD is provided in grayscale). Following standard practice on this benchmark, we evaluate at the **frame level** (binary: anomalous vs. normal frame) using the provided ground-truth annotations, and treat anomaly detection as binary classification rather than localization. Pixel-level evaluation is treated as a stretch goal.

### 3.2 RPCA Preprocessing

For each video clip, frames are vectorized and stacked column-wise into a matrix $X \in \mathbb{R}^{(H \cdot W) \times T}$, where $T$ is the number of frames in the clip. We apply the inexact ALM solver (using $\mu = n_1 n_2 / (4\|X\|_1)$ and $\lambda = 1/\sqrt{\max(n_1, n_2)}$, following Candès et al., 2011) to obtain the decomposition $X = L + S$. The resulting $L$ and $S$ matrices are reshaped back to frame sequences for downstream processing. RPCA is applied per-clip rather than globally, since each clip has its own static background and a single global decomposition would lose the per-scene statistical structure.

Convergence behavior, computational cost, and qualitative L/S decompositions on representative clips will be documented as part of the results.

### 3.3 Classification Architectures

We compare four input representations on a fixed CNN backbone (ResNet-18, pretrained on ImageNet, with the input layer adapted for grayscale):

1. **Baseline:** Raw grayscale frames
2. **L-only:** Low-rank background component
3. **S-only:** Sparse foreground component
4. **[L, S]:** Concatenated as a 2-channel input

Frame-level prediction is performed independently per frame (no temporal model), keeping the architecture comparable across input types and isolating the contribution of the RPCA preprocessing step. If time permits, a small temporal-context extension (e.g., stacking three consecutive frames) will be evaluated as an ablation.

### 3.4 Evaluation Protocol

Following standard practice on UCSD, we report **frame-level AUC** on the test set as the primary metric, with ROC curves and Equal Error Rate (EER) as secondary metrics. We additionally report:
- Inference time (RPCA preprocessing + forward pass) to characterize the practical cost of the proposed pipeline
- Qualitative visualization of L/S decompositions on representative anomalous and normal clips, including failure cases
- Confusion analysis distinguishing the major UCSD anomaly types (cyclist, skateboarder, vehicle, cart)

### 3.5 Comparison with Public Benchmarks

UCSD Ped1 and Ped2 are heavily benchmarked. Reference points from the literature include:
- **Mahadevan et al. (2010), MDT:** ~81.8% AUC on Ped1, ~82.9% AUC on Ped2 — the original benchmark
- **Hasan et al. (2016), Conv-AE:** ~81.0% AUC on Ped1, ~90.0% AUC on Ped2 — early deep-learning method
- **Recent SOTA methods:** typically 95-97% AUC on Ped2

Our goal is *not* to achieve state-of-the-art performance, but to isolate the contribution of RPCA preprocessing as an interpretable, classical-method augmentation to a standard deep pipeline. A favorable outcome includes any of:
1. RPCA preprocessing improves AUC over the matched baseline
2. RPCA preprocessing achieves competitive AUC with a simpler / faster classifier than end-to-end deep methods
3. A clear negative result with diagnostic explanation of *why* RPCA fails on this data — itself a valuable finding

Specific reference numbers will be re-verified against primary sources in the final report.

## 4. Connection to Course Material

This project synthesizes three core methods from the course:
- **SVD** appears explicitly in the singular value thresholding (SVT) operator at the heart of the RPCA solver, and again in the optional model-compression ablation (RQ4)
- **RPCA** is the central preprocessing operation, and we treat its empirical behavior on a real benchmark as a contribution in itself
- **Compressed sensing intuition** motivates the project: the assumption that anomalous content is *sparse in the pixel domain* (after background removal) is a direct analog of the sparsity-in-a-basis assumption underlying compressed sensing reconstruction

## 5. Expected Deliverables

1. A trained pipeline (RPCA + CNN) with reproducible code
2. A written report covering motivation, methodology, results, and discussion
3. A presentation summarizing key findings, including qualitative L/S decomposition visualizations
4. A quantitative comparison table positioning our results against published UCSD benchmarks

## 6. Risks and Mitigation

- **Compute risk:** RPCA is iterative and operates on full-resolution video matrices. *Mitigation:* If runtime is excessive, frames can be downsampled spatially before decomposition.
- **Small dataset size:** UCSD has very limited training data, especially Ped2. *Mitigation:* Use ImageNet-pretrained backbone, aggressive data augmentation, and frame-level rather than clip-level prediction to maximize effective sample count. Report AUC (insensitive to class imbalance) rather than accuracy.
- **Time risk:** Three days is tight. *Mitigation:* Prioritize RQ1 on Ped2 (smaller, easier, faster to iterate) as the minimum viable result. Extend to Ped1 and ablations only after the primary result is established.