# Falsification Report: PWODE V.8 (2D Image Analysis)

**Document Version:** 1.0 (Public Release)  
**Date:** [Insert Current Date: e.g., 2025-10-29]  
**Status:** Falsified. Basis for scientific pivot (Path C).  
**Code Reference:** `[./code/Legacy_2D_V8/]`

---

## 1. Executive Summary: The Failure and Pivot

The initial application of the Prime Wave Order-Detection Engine (PWODE) hypothesized that prime-inspired arithmetic filtering could detect coherent order in dense 2D image data.

**This hypothesis was falsified.** Rigorous benchmarking against industry standards (SIFT/FAST) confirmed that the 2D pipeline failed to effectively discriminate signal from noise, yielding a performance metric indistinguishable from random guessing within the test domain.

The measured performance metric was: **Precision $\approx 3.9\%$ on average.**

This failure mandated the **Path C Pivot** from high-dimensional, dense data to sparse 1D Quantum Energy Spectra, where the core hypothesis has since been successfully validated (PWODE V.9.3).

---

## 2. Methodological Failure Analysis

The PWODE V.8 (2D) pipeline utilized a three-phase cascade which ultimately collapsed when faced with the high correlation and redundancy inherent in dense image data.

**Key Failures of the 2D Architecture:**

| Component | Diagnosis of Failure |
|-----------|----------------------|
| **Data Domain** | Dense/Non-Sparse Data. The density of information in 2D arrays exceeded the limit required by the theoretical filter constraint ($d < 1/\text{primorial}(N)$), leading to filter overload. |
| **Arithmetic Function** | Quadratic Coherence Mapping ($\mathbf{i^2}$) was used to predict feature echoes. This function was found to be ill-suited for the arithmetic structure of 2D coordinates. |
| **Discrimination** | The Quadratic Coherence Score (QCS) failed to reliably differentiate ordered signal from stochastic noise, resulting in an unacceptable number of false positives (or near-zero detection). |

---

## 3. Quantitative Evidence of Failure (V.8 Benchmark)

The following data is extracted from the benchmark suite run (testing PWODE V.8.3 against SIFT and FAST on 600 synthetic images across varying noise levels).

**Benchmark Summary (Average Metrics Across All Noise Levels)**

The critical metric for feature detection is **Precision** (True Positives / All Detections).

| Detector | Average Precision | Average F1-Score | Average Runtime (s) |
|----------|-------------------|------------------|---------------------|
| SIFT (Baseline) | $0.634 \pm 0.10$ | $0.696 \pm 0.07$ | $0.0003$ |
| FAST (Speed Baseline) | $0.320 \pm 0.06$ | $0.482 \pm 0.07$ | $0.0000$ |
| **PWODE (V.8.3)** | $\mathbf{0.039 \pm 0.002}$ | $\mathbf{0.075 \pm 0.003}$ | $0.0209$ |

**Interpretation of PWODE Results:**

- **Precision Failure:** The average precision of $\mathbf{0.039}$ means that less than $4\%$ of the features flagged by PWODE were true positives (matching the ground truth). This confirms that the model was unable to detect coherent features in the 2D domain.
- **Trade-Off Collapse:** The output shows a negative trade-off versus SIFT: "Trade-off: 0.0x faster with -59.4% precision." This confirms the 2D approach provided neither speed nor accuracy.

---

## 4. Scientific Conclusion

The V.8 results demonstrate that a $\mathbf{PNT}$ filter applied naively to a generic, dense domain collapses. This failure was crucial, forcing the project to seek a domain where the hypothesis—that prime-inspired arithmetic order aligns with non-random physical structure—could be tested under scientifically sound, sparse conditions.

The successful $\mathbf{V.9.3}$ pivot to 1D VDOS, with the discovery of the superior $\mathbf{i \cdot \ln(i)}$ echo function, is the direct and validated scientific outcome of this falsification.
