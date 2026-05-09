"""LLNL replication of Wang/Khor/Desnoyers 2DIO trace-generation baseline.

Source paper: Wang, Khor, Desnoyers, "2DIO: Configurable and Cache-Accurate
Trace Generation for Storage Benchmarking", EUROSYS 2026.
PDF: pubs/2DIO_CacheAccurate_2026.pdf

Modules:
- irm.py     : the paper's vanilla-IRM weak baseline (Figure 3 "IRM-recon").
               Pure i.i.d. sampling from the real trace's frequency PMF.
               Re-implemented here for sanity-checking the heap-based 2DIO.
- two_dio.py : faithful re-implementation of the paper's Algorithm 2
               (Gen-from-2D). Heap scheduler over IRDs sampled from
               fgen(k, I, ε) (paper Eq. 3) plus an IRM-arrival channel
               with probability P_IRM, driven by item-frequency
               distribution g (Table 2 supports Zipf/Pareto/Normal/
               Uniform/Empirical). The t=∞ branch of Algorithm 2 is
               controlled via the --p-inf parameter (per-draw probability
               of one-shot from the IRD channel; the paper's fgen has
               finite support so this mass enters separately).

CLI:
    python -m desnoyers.two_dio --p-irm <p> --g <gspec> --f <fgen-spec> \\
                                --p-inf <p_inf> -m <M> -n <N> --seed <S> \\
                                --output <fake.csv>

INDEXING NOTE: Table 3 fgen I-values are 0-INDEXED (verified against
v766 [0,5] and v827 [0,13] entries that contain 0 — invalid under
1-indexed support {1..k}). Pass them verbatim to --f.
"""
