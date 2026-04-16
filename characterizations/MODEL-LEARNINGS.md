# Model Learnings

This report joins the family characterization pass with the full train/eval history from `vinge.local`.

## Inventory

- Train logs parsed: 250
- Eval logs parsed: 61

## Tencent

| Item | Value |
|---|---|
| Evaluated best checkpoint | tencent_v103 |
| Evaluated best combined score | 0.098 |
| Best evaluated recall checkpoint | tencent_v93 |
| Best evaluated recall | 0.553 |
| Frontier train-only checkpoint | tencent_v74 |
| Frontier best train combined | 0.065 |

### Validated Lessons

- PCF-on evaluated runs shift combined score by -0.01 relative to PCF-off for this corpus.
- Block sampling shifts evaluated combined score by 0 relative to random sampling.

### Frontier Lessons

- Block sampling shifts best-train combined by -0.004 in the running history.
- PCF-on shifts best-train combined by -0.006 in the running history.
- Multi-scale critic shifts best-train combined by -0.022 in the running history.
- Mixed-type recovery shifts best-train combined by 0 in the running history.
- Retrieval memory shifts best-train combined by -0.036 in the running history.

### Top Evaluated Checkpoints

| Run | Combined | Recall | MMD² | DMD-GEN | HRC-MAE |
|---|---:|---:|---:|---:|---:|
| tencent_v103 | 0.098 | 0.547 | 0.008 | 0.737 | 0.06 |
| tencent_v93 | 0.099 | 0.553 | 0.01 | 0.685 | 0.025 |
| tencent_v78 | 0.101 | 0.532 | 0.007 | 0.742 | 0.08 |
| tencent_v88 | 0.11 | 0.516 | 0.013 | 0.736 | 0.008 |
| tencent_v76 | 0.112 | 0.537 | 0.02 | 0.701 | 0.076 |
| tencent_v99 | 0.112 | 0.507 | 0.014 | 0.718 | 0 |
| tencent_v86 | 0.113 | 0.521 | 0.017 | 0.714 | 0.049 |
| tencent_v100 | 0.12 | 0.486 | 0.017 | 0.705 | 0.049 |

### Frontier Train Runs

| Run | Best Train Combined | Recall | MMD² |
|---|---:|---:|---:|
| tencent_v74 | 0.065 | 0.7 | 0.005 |
| tencent_v136 | 0.073 | 0.694 | 0.012 |
| tencent_v66 | 0.077 | 0.651 | 0.007 |
| tencent_v143 | 0.077 | 0.656 | 0.008 |
| tencent_v144 | 0.082 | 0.631 | 0.008 |
| tencent_v82 | 0.082 | 0.621 | 0.006 |
| tencent_v137 | 0.082 | 0.628 | 0.008 |
| tencent_v134 | 0.083 | 0.629 | 0.008 |

## Alibaba

| Item | Value |
|---|---|
| Evaluated best checkpoint | alibaba_v71 |
| Evaluated best combined score | 0.067 |
| Best evaluated recall checkpoint | alibaba_v71 |
| Best evaluated recall | 0.701 |
| Frontier train-only checkpoint | alibaba_v116 |
| Frontier best train combined | 0.069 |

### Validated Lessons

- PCF-on evaluated runs shift combined score by 0.009 relative to PCF-off for this corpus.
- Block sampling shifts evaluated combined score by 0 relative to random sampling.

### Frontier Lessons

- Block sampling shifts best-train combined by -0.044 in the running history.
- PCF-on shifts best-train combined by -0.034 in the running history.
- Multi-scale critic shifts best-train combined by -0.019 in the running history.
- Mixed-type recovery shifts best-train combined by -0.029 in the running history.

### Top Evaluated Checkpoints

| Run | Combined | Recall | MMD² | DMD-GEN | HRC-MAE |
|---|---:|---:|---:|---:|---:|
| alibaba_v71 | 0.067 | 0.701 | 0.007 | 0.702 | 0.01 |
| alibaba_v48 | 0.077 | 0.681 | 0.013 | 0.737 | 0.001 |
| alibaba_v37 | 0.079 | 0.664 | 0.011 | 0.759 | 0.023 |
| alibaba_v74 | 0.093 | 0.613 | 0.015 | 0.702 | 0.006 |
| alibaba_v38 | 0.094 | 0.614 | 0.017 | 0.745 | 0.002 |
| alibaba_v72 | 0.111 | 0.535 | 0.018 | 0.705 | 0.009 |
| alibaba_v59 | 0.111 | 0.526 | 0.016 | 0.769 | 0.01 |
| alibaba_v57 | 0.113 | 0.501 | 0.014 | 0.793 | 0.006 |

### Frontier Train Runs

| Run | Best Train Combined | Recall | MMD² |
|---|---:|---:|---:|
| alibaba_v116 | 0.069 | 0.703 | 0.01 |
| alibaba_v51 | 0.071 | 0.679 | 0.007 |
| alibaba_v114 | 0.073 | 0.694 | 0.012 |
| alibaba_v87 | 0.077 | 0.681 | 0.013 |
| alibaba_v79 | 0.079 | 0.663 | 0.011 |
| alibaba_v88 | 0.08 | 0.648 | 0.01 |
| alibaba_v115 | 0.083 | 0.654 | 0.014 |
| alibaba_v111 | 0.083 | 0.661 | 0.016 |

