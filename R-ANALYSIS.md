# R Analysis

## Scope

This document describes the math used by the R characterization pipeline.

Important boundary:

- R did not decode raw traces.
- Python decoded raw traces, computed per-file profiles, and wrote normalized JSONL.
- R consumed those normalized per-file profiles and did family-level statistics, multivariate analysis, regime analysis, outlier analysis, and report generation.

Main entrypoints:

- [extract_family_features.R](/Users/darrell/Zarathustra/R-scripts/extract_family_features.R)
- [analyze_family.R](/Users/darrell/Zarathustra/R-scripts/analyze_family.R)
- [render_family_report.R](/Users/darrell/Zarathustra/R-scripts/render_family_report.R)
- [run_corpus_analysis.R](/Users/darrell/Zarathustra/R-scripts/run_corpus_analysis.R)
- [run_model_aware_analysis.R](/Users/darrell/Zarathustra/R-scripts/run_model_aware_analysis.R)

## Data Flow

1. Python wrote normalized per-file rows into `trace_characterizations.normalized.jsonl`.
2. R flattened each JSON row into one tabular row of scalar features.
3. R grouped rows by
   \( \text{logical\_family\_id} = \text{"dataset\_\_family"} \),
   where the string is formed by concatenating `dataset`, `"__"`, and `family`.
4. R ran one family-level analysis per group.
5. R wrote `analysis.json`, `metric_summary.csv`, optional diagnostics CSVs, and one Markdown report per family.

## Notation

For a family with \( n \) files and \( p \) numeric features:

- \( x_{ij} \) is feature \( j \) for file \( i \)
- \( x_{\cdot j} \) is the vector of all finite values for feature \( j \)
- \( m_j \) is the number of finite observations for feature \( j \)
- \( \widetilde{x}_j \) is the sample median of feature \( j \)
- \( s_j \) is the sample standard deviation of feature \( j \)

When a feature is only partially present, R computes the statistic on the finite subset.

## Parser-Side Math Passed Into R

These formulas were computed in Python before R ever saw the data.

### Request-Sequence Profiles

Given timestamps \( t_1, \dots, t_n \), opcodes \( o_1, \dots, o_n \), object ids \( u_1, \dots, u_n \), and object sizes \( b_1, \dots, b_n \):

Inter-arrival times:

$$
\mathrm{iat}_i = \max(0, t_i - t_{i-1}), \quad i = 2, \dots, n
$$

Duration:

$$
\mathrm{ts\_duration} = t_n - t_1
$$

Sample record rate:

$$
\mathrm{sample\_record\_rate} =
\begin{cases}
\dfrac{n}{\mathrm{ts\_duration}}, & \mathrm{ts\_duration} > 0 \\
\mathrm{NA}, & \text{otherwise}
\end{cases}
$$

Zero-IAT ratio:

$$
\mathrm{iat\_zero\_ratio} =
\dfrac{1}{n-1} \sum_{i=2}^{n} \mathbf{1}\{ \mathrm{iat}_i = 0 \}
$$

Burstiness coefficient of variation:

$$
\mathrm{burstiness\_cv} =
\begin{cases}
\dfrac{\mathrm{sd}(\mathrm{iat})}{\mathrm{mean}(\mathrm{iat})}, & \mathrm{mean}(\mathrm{iat}) \neq 0 \\
\mathrm{NA}, & \text{otherwise}
\end{cases}
$$

Write ratio:

$$
\mathrm{write\_ratio} = \dfrac{1}{n} \sum_{i=1}^{n} \mathbf{1}\{ o_i < 0 \}
$$

Opcode switch ratio:

$$
\mathrm{opcode\_switch\_ratio} =
\begin{cases}
\dfrac{1}{n-1} \sum_{i=2}^{n} \mathbf{1}\{ o_i \neq o_{i-1} \}, & n \geq 2 \\
\mathrm{NA}, & \text{otherwise}
\end{cases}
$$

Signed object-id stride:

$$
\Delta_i = u_i - u_{i-1}, \quad i = 2, \dots, n
$$

Reuse, forward-seek, and backward-seek ratios:

$$
\mathrm{reuse\_ratio} =
\dfrac{1}{n-1} \sum_{i=2}^{n} \mathbf{1}\{ \Delta_i = 0 \}
$$

$$
\mathrm{forward\_seek\_ratio} =
\dfrac{1}{n-1} \sum_{i=2}^{n} \mathbf{1}\{ \Delta_i > 0 \}
$$

$$
\mathrm{backward\_seek\_ratio} =
\dfrac{1}{n-1} \sum_{i=2}^{n} \mathbf{1}\{ \Delta_i < 0 \}
$$

Lag-1 autocorrelation for a sequence \( y_1, \dots, y_m \):

$$
\rho_1(y) =
\dfrac{\sum_{k=2}^{m} (y_k - \overline{y})(y_{k-1} - \overline{y})}
{\sum_{k=1}^{m} (y_k - \overline{y})^2}
$$

This was used for `iat_lag1_autocorr` and `signed_stride_lag1_autocorr` when the denominator was nonzero.

Summary fields like `iat_q50`, `iat_q90`, `iat_q99`, `obj_size_q50`, and `abs_stride_q90` were empirical quantiles of those derived vectors.

### Aggregate Time-Series Profiles

Given timestamped samples with read IOPS \( r_i \), write IOPS \( w_i \), read bandwidth \( R_i \), write bandwidth \( W_i \), and disk usage \( d_i \):

Sampling interval:

$$
\delta_i = \max(0, t_i - t_{i-1}), \quad i = 2, \dots, n
$$

Total IOPS:

$$
\mathrm{total\_iops}_i = r_i + w_i
$$

Total bandwidth:

$$
\mathrm{total\_bw}_i = R_i + W_i
$$

Idle ratio:

$$
\mathrm{idle\_ratio} = \dfrac{1}{n} \sum_{i=1}^{n} \mathbf{1}\{ \mathrm{total\_iops}_i = 0 \}
$$

Write-share IOPS mean:

$$
\mathrm{write\_share\_iops\_mean} =
\begin{cases}
\dfrac{\sum_{i=1}^{n} w_i}{\sum_{i=1}^{n} (r_i + w_i)}, & \sum_{i=1}^{n} (r_i + w_i) > 0 \\
\mathrm{NA}, & \text{otherwise}
\end{cases}
$$

### Structured-Table Profiles

Given schema column summaries \( c_1, \dots, c_q \):

Column count:

$$
\mathrm{schema\_column\_count} = q
$$

Numeric-column count:

$$
\mathrm{schema\_numeric\_cols} =
\sum_{\ell=1}^{q} \mathbf{1}\{ \mathrm{numeric\_ratio}(c_\ell) \geq 0.8 \}
$$

Mixed-column count:

$$
\mathrm{schema\_mixed\_cols} =
\sum_{\ell=1}^{q} \mathbf{1}\{ 0.2 < \mathrm{numeric\_ratio}(c_\ell) < 0.8 \}
$$

High-cardinality-column count:

$$
\mathrm{schema\_high\_cardinality\_cols} =
\sum_{\ell=1}^{q} \mathbf{1}\{ \mathrm{unique\_tokens}(c_\ell) \geq 100 \}
$$

## R Flattening

`extract_family_features.R` performs deterministic field extraction only:

- missing paths become `NA`
- boolean flags become \( 0 \) or \( 1 \)
- counts and summaries are copied into one wide row per file

No estimation or learned model enters here.

## Univariate Statistics Per Feature

For each feature \( j \), R takes the finite subset

$$
x_{\cdot j}^{\ast} = \{ x_{ij} : x_{ij} \text{ is finite} \}
$$

and computes:

Finite count:

$$
m_j = \left| x_{\cdot j}^{\ast} \right|
$$

Missing fraction:

$$
\mathrm{missing\_frac}_j =
1 - \dfrac{m_j}{n}
$$

Mean:

$$
\overline{x}_j = \dfrac{1}{m_j} \sum_{x \in x_{\cdot j}^{\ast}} x
$$

Median:

$$
\widetilde{x}_j = \mathrm{median}(x_{\cdot j}^{\ast})
$$

Sample standard deviation:

$$
s_j =
\sqrt{
\dfrac{1}{m_j - 1}
\sum_{x \in x_{\cdot j}^{\ast}} (x - \overline{x}_j)^2
}
$$

Median absolute deviation with R's `constant = 1`:

$$
\mathrm{MAD}_j = \mathrm{median}\left( \left| x - \widetilde{x}_j \right| : x \in x_{\cdot j}^{\ast} \right)
$$

Coefficient of variation:

$$
\mathrm{CV}_j =
\begin{cases}
\dfrac{s_j}{\left| \overline{x}_j \right|}, & \left| \overline{x}_j \right| > 10^{-12} \\
\mathrm{NA}, & \text{otherwise}
\end{cases}
$$

Sample skewness and kurtosis came from package implementations:

- `e1071::skewness(, type = 2)` when available, else `moments::skewness`
- `e1071::kurtosis(, type = 2)` when available, else `moments::kurtosis`

### Quantiles

R used `stats::quantile(, type = 7)`. For probability \( p \in (0,1) \), let the sorted sample be
\( x_{(1)} \leq \dots \leq x_{(m)} \), and define

$$
h = (m - 1)p + 1
$$

$$
k = \lfloor h \rfloor, \quad \gamma = h - k
$$

Then the type-7 quantile is

$$
Q(p) = (1 - \gamma) x_{(k)} + \gamma x_{(k+1)}
$$

with the obvious endpoint handling when \( h \) lands on the boundary.

R computed \( Q(0.10) \), \( Q(0.25) \), \( Q(0.75) \), and \( Q(0.90) \) for the family reports, and the parser-side Python code had already supplied many \( Q(0.50) \), \( Q(0.90) \), and \( Q(0.99) \) summaries per file.

## Heterogeneity Score

For a family with valid per-feature CVs \( \mathrm{CV}_1, \dots, \mathrm{CV}_r \), R defines

$$
\mathrm{heterogeneity\_score} =
\mathrm{median}\left(
\min(\mathrm{CV}_1, 25),
\dots,
\min(\mathrm{CV}_r, 25)
\right)
$$

This is a robust cross-feature dispersion summary:

- large when many features vary strongly across files
- small when files are similar across the available feature surface

## Feature Matrix For Multivariate Analysis

Starting from the wide feature table, R builds the multivariate matrix in five steps:

1. keep only `numeric_columns`
2. coerce every retained column to numeric
3. keep column \( j \) only if it has at least
   $$
   \max(4, \lfloor 0.5 n \rfloor)
   $$
   finite values
4. keep only rows that are complete across the retained columns
5. drop zero-variance columns

So PCA and clustering operate on a stricter complete-case matrix than the univariate summaries.

## Correlation Analysis

R computes the pairwise-complete Pearson correlation matrix

$$
C = \mathrm{cor}(X, \text{use} = \text{"pairwise.complete.obs"})
$$

For each feature pair \( (a,b) \), it records

$$
\left| C_{ab} \right|
$$

and ranks pairs by that absolute correlation.

## PCA

Let the complete-case matrix be \( X \in \mathbb{R}^{n_c \times p_c} \).

R centers and scales each column:

$$
z_{ij} = \dfrac{x_{ij} - \mu_j}{\sigma_j}
$$

where \( \mu_j \) and \( \sigma_j \) are the column mean and sample standard deviation on the complete-case matrix.

Then `prcomp(..., center = TRUE, scale. = TRUE)` computes the singular value decomposition of the standardized matrix. Equivalently, PCA diagonalizes the sample covariance of \( Z \):

$$
\dfrac{1}{n_c - 1} Z^\top Z = V \Lambda V^\top
$$

Outputs used by the reports:

- principal-component scores
  $$
  S = ZV
  $$
- loadings \( V \)
- variance explained by each component
- `pc1_variance`, the proportion explained by PC1

## K-Means Diagnostics And Final Fit

R evaluates

$$
k \in \{ 2, 3, \dots, \min(12, n_c - 1) \}
$$

For each \( k \), it fits k-means with `nstart = 10` and records total within-cluster sum of squares:

$$
\mathrm{WSS}(k) =
\sum_{g=1}^{k} \sum_{i \in C_g}
\left\| s_i - \mu_g \right\|_2^2
$$

where \( s_i \) is the feature vector or PCA-space vector being clustered and \( \mu_g \) is cluster centroid \( g \).

If the `cluster` package is available, R also computes the average silhouette width. For file \( i \):

$$
a(i) = \text{mean dissimilarity from } i \text{ to its own cluster}
$$

$$
b(i) = \min_{g \neq c(i)} \text{mean dissimilarity from } i \text{ to cluster } g
$$

$$
\mathrm{sil}(i) =
\dfrac{b(i) - a(i)}{\max(a(i), b(i))}
$$

and the reported silhouette score is

$$
\overline{\mathrm{sil}}(k) =
\dfrac{1}{n_c} \sum_{i=1}^{n_c} \mathrm{sil}(i)
$$

If \( n_c > 1200 \), silhouette scoring is evaluated on a random 1200-row subset for cost control.

Selected \( k \):

$$
k^\ast =
\begin{cases}
\arg\max_k \overline{\mathrm{sil}}(k), & \text{if any finite silhouette exists} \\
2, & \text{otherwise}
\end{cases}
$$

Final k-means fit:

$$
\text{kmeans}(X, \text{centers} = k^\ast, \text{nstart} = 20)
$$

## Gaussian Mixture Model

R runs `mclust::Mclust` over

$$
G \in \{ 1, 2, \dots, \min(6, n_c - 1) \}
$$

and keeps the model with maximum BIC.

For a candidate model with log-likelihood \( \ell \), \( d \) free parameters, and \( n_c \) observations, the BIC score used by `mclust` is of the form

$$
\mathrm{BIC} = 2 \ell - d \log n_c
$$

The pipeline retains:

- selected component count \( G^\ast \)
- model name
- maximum BIC
- mean classification uncertainty

If posterior responsibilities are \( \tau_{ig} \), then per-file uncertainty is

$$
u_i = 1 - \max_g \tau_{ig}
$$

and the report stores

$$
\overline{u} = \dfrac{1}{n_c} \sum_{i=1}^{n_c} u_i
$$

## DBSCAN

R sets

$$
\mathrm{minPts} = \max\left(4, \min\left(20, \left\lfloor 3 \log n_c \right\rfloor \right)\right)
$$

It then computes the `kNNdist` vector using \( \mathrm{minPts} \) neighbors and sets

$$
\varepsilon = Q_{0.90}(\mathrm{kNNdist})
$$

Then it runs DBSCAN with that \( \varepsilon \) and `minPts`.

Reported outputs:

- number of non-noise clusters
- noise fraction
  $$
  \mathrm{noise\_fraction} =
  \dfrac{1}{n_c} \sum_{i=1}^{n_c} \mathbf{1}\{ \mathrm{cluster}_i = 0 \}
  $$

## Outlier Scoring

Let \( S \) be the retained PCA-score matrix after dropping zero-variance PCs.

Primary outlier score:

$$
d_i^2 = (s_i - \mu)^\top \Sigma^{-1} (s_i - \mu)
$$

where \( \mu \) is the mean PCA-score vector and \( \Sigma \) is the sample covariance of the PCA scores.

This is Mahalanobis distance squared.

Fallback if covariance inversion fails:

$$
\mathrm{outlier\_score}_i =
\sum_{j=1}^{q}
\left(
\dfrac{s_{ij} - \overline{s}_{\cdot j}}
{\mathrm{sd}(s_{\cdot j})}
\right)^2
$$

R ranks files by descending outlier score and keeps the top 8 in the report.

## Outlier Decomposition

For each numeric feature \( j \), R computes a robust center and scale:

$$
c_j = \mathrm{median}(x_{\cdot j})
$$

$$
r_j = \mathrm{MAD}(x_{\cdot j})
$$

Fallbacks:

$$
r_j =
\begin{cases}
\mathrm{MAD}(x_{\cdot j}), & \mathrm{MAD}(x_{\cdot j}) > 10^{-9} \\
\mathrm{sd}(x_{\cdot j}), & \mathrm{sd}(x_{\cdot j}) > 10^{-9} \\
1, & \text{otherwise}
\end{cases}
$$

Then for each reported outlier file \( i \), R computes robust per-feature z-scores:

$$
z_{ij} = \dfrac{x_{ij} - c_j}{r_j}
$$

It ranks features by \( |z_{ij}| \) and reports the top contributors.

## Outlier Sensitivity

For the metric set

- `burstiness_cv`
- `abs_stride_mean`
- `obj_size_std`
- `reuse_ratio`
- `iat_q50`
- `object_unique`

and for

$$
N \in \{ 1, 3, 5, 10 \}
$$

when valid for the family, R removes the top \( N \) outliers and compares medians.

Baseline median:

$$
m_j^{\mathrm{base}} = \mathrm{median}(x_{\cdot j})
$$

Trimmed median:

$$
m_{j,N}^{\mathrm{trim}} =
\mathrm{median}(x_{\cdot j} \setminus \text{top-}N\text{ outliers})
$$

Relative shift:

$$
\mathrm{relative\_shift}_{j,N} =
\dfrac{m_{j,N}^{\mathrm{trim}} - m_j^{\mathrm{base}}}
{\max\left( \left| m_j^{\mathrm{base}} \right|, 10^{-9} \right)}
$$

Rows are ranked by \( |\mathrm{relative\_shift}_{j,N}| \).

## Regime Detection

R attempts regime detection only when:

- `ts_start` exists
- PCA exists
- at least 8 ordered files exist

Files are ordered by

$$
( \mathrm{ts\_start}, \mathrm{rel\_path} )
$$

and R takes the ordered PC1 sequence

$$
y_1, y_2, \dots, y_m
$$

It then runs

$$
\texttt{changepoint::cpt.meanvar}(y, \texttt{method = "PELT"}, \texttt{penalty = "SIC"})
$$

Conceptually, this chooses changepoints that minimize penalized segment cost:

$$
\sum_{r=1}^{R} \mathrm{cost}(y_{\tau_{r-1}+1:\tau_r}) + \beta R
$$

with PELT search and SIC-style penalty.

R records:

- changepoint count
- changepoint indices

R also calls `tsfeatures::tsfeatures` on the ordered PC1 series and stores package-defined scalar summaries such as entropy, lumpiness, stability, Hurst, and flat-spots. Those are library outputs, not formulas reimplemented in this repo.

## Regime Attribution

If changepoints are

$$
1 \leq c_1 < c_2 < \dots < c_m < n
$$

then segments are

- segment 1: \( 1, \dots, c_1 \)
- segment 2: \( c_1 + 1, \dots, c_2 \)
- ...
- segment \( m+1 \): \( c_m + 1, \dots, n \)

For each segment \( r \), R stores:

- file count
- minimum and maximum `ts_start`
- \( \mathrm{median}(\mathrm{PC1}) \)
- \( \mathrm{sd}(\mathrm{PC1}) \)

For each adjacent segment pair \( A \) and \( B \), and each feature \( j \):

Left and right medians:

$$
a_j = \mathrm{median}(x_{A,j}), \quad b_j = \mathrm{median}(x_{B,j})
$$

Robust scales:

$$
s_{A,j} = \mathrm{MAD}(x_{A,j}), \quad s_{B,j} = \mathrm{MAD}(x_{B,j})
$$

Pooled robust scale:

$$
s_{p,j} = \sqrt{\dfrac{s_{A,j}^2 + s_{B,j}^2}{2}}
$$

Fallback if \( s_{p,j} \) is zero or non-finite:

$$
s_{p,j} =
\sqrt{
\dfrac{\mathrm{var}(x_{A,j}) + \mathrm{var}(x_{B,j})}{2}
}
$$

Transition effect size:

$$
e_j = \dfrac{|a_j - b_j|}{s_{p,j}}
$$

For each transition, features are ranked by \( e_j \), and the top drivers are written into `regime_transitions.csv` and into the family report.

## Temporal Sampling Diagnostic

To test whether sequential blocks are more coherent than random batches, R:

1. orders files by `ts_start`
2. keeps available PC coordinates among PC1, PC2, and PC3
3. chooses
   $$
   \mathrm{block\_size} =
   \max\left(4, \min\left(16, \left\lfloor \sqrt{n} \right\rfloor \right)\right)
   $$
4. splits the ordered series into contiguous non-overlapping blocks
5. computes the mean pairwise Euclidean distance inside each block
6. draws 64 random batches of the same size and computes the same statistic

For a block \( B \), the within-block distance summary is

$$
D(B) =
\dfrac{1}{|P(B)|}
\sum_{(u,v) \in P(B)}
\left\| s_u - s_v \right\|_2
$$

where \( P(B) \) is the set of unordered pairs inside the block.

R then computes

$$
\mathrm{median\_block\_distance} = \mathrm{median}(D(B_1), \dots, D(B_q))
$$

$$
\mathrm{median\_random\_distance} = \mathrm{median}(D(R_1), \dots, D(R_{64}))
$$

$$
\mathrm{block\_random\_distance\_ratio} =
\dfrac{\mathrm{median\_block\_distance}}
{\mathrm{median\_random\_distance}}
$$

Interpretation:

- values near \( 1 \) mean ordered blocks are not much more coherent than random batches
- values well below \( 1 \) mean temporal adjacency matters

The current report emits a block-sampling warning when the ratio is below \( 0.85 \).

## Conditioning Audit

R audits the current GAN-facing conditioning set:

- `write_ratio`
- `reuse_ratio`
- `burstiness_cv`
- `iat_q50`
- `obj_size_q50`
- `opcode_switch_ratio`
- `iat_lag1_autocorr`
- `tenant_unique`
- `forward_seek_ratio`
- `backward_seek_ratio`

### Near-Constant Detection

A current conditioning feature \( j \) is flagged as near-constant when either

$$
\mathrm{sd}(x_{\cdot j}) \leq 10^{-9}
$$

or

$$
\mathrm{IQR}(x_{\cdot j}) \leq 10^{-9}
$$

### Redundancy Detection

Using the family correlation matrix \( C \), a pair \( (i,j) \) is flagged as highly redundant when

$$
|C_{ij}| \geq 0.95
$$

### Candidate Additions

R separately evaluates:

- `object_unique`
- `signed_stride_lag1_autocorr`
- `obj_size_std`

For each candidate \( z \), it stores:

- \( \mathrm{sd}(z) \)
- \( \mathrm{IQR}(z) \)
- maximum absolute correlation with the current conditioning set
  $$
  \max_j | \mathrm{corr}(z, x_j) |
  $$

A candidate is marked recommended when

$$
\mathrm{IQR}(z) > 0
$$

and

$$
\max_j | \mathrm{corr}(z, x_j) | < 0.95
$$

or when that maximum correlation is unavailable.

## Suggested GAN Modes

R defines

$$
\mathrm{mclust\_modes} =
\begin{cases}
G^\ast, & \text{if Mclust succeeded} \\
1, & \text{otherwise}
\end{cases}
$$

$$
\mathrm{regime\_modes} =
\begin{cases}
\mathrm{changepoint\_count} + 1, & \text{if changepoints exist} \\
1, & \text{otherwise}
\end{cases}
$$

Then

$$
\mathrm{suggested\_modes} =
\max\left(
1,
\min(8, \mathrm{mclust\_modes}),
\min(8, \mathrm{regime\_modes})
\right)
$$

So the recommendation is explicitly capped at \( 8 \).

## Split-By-Format Flag

R sets

$$
\mathrm{split\_by\_format} =
\mathbf{1}\{
\# \mathrm{formats} > 1
\text{ or }
\# \mathrm{parsers} > 1
\}
$$

This is a hard warning that the family should not be pooled naively.

## Rule-Based GAN Guidance

The report guidance bullets are deterministic rules on top of the computed statistics. Examples:

- if `split_by_format = TRUE`, warn to split by encoding
- if `heterogeneity_score > 2.5`, warn against one unconditional model
- if Mclust finds at least 3 components, note multi-mode structure
- if changepoints exist, note regime structure
- if mean read share exceeds \( 0.9 \), warn that the family is strongly read-skewed
- if mean write ratio exceeds \( 0.6 \), emphasize write-pressure preservation
- if median `reuse_ratio` exceeds \( 0.5 \), emphasize locality-aware conditioning
- if median `burstiness_cv` exceeds \( 10 \), emphasize burst-sensitive losses
- if median `tenant_unique` exceeds \( 8 \), suggest tenant-aware conditioning
- if `block_random_distance_ratio < 0.85`, suggest block or curriculum sampling
- if the conditioning audit recommends additions, mention those candidate features

No learned classifier or optimizer chooses these text bullets.

## Model-Aware R Layer

The newer pass adds a second R analysis that consumes:

- family rollups from the characterization pass
- train logs from `/home/darrell/train_*.log`
- eval logs from `/home/darrell/eval_*.log`

This layer does not retrain models. It summarizes observed behavior from prior Tencent and Alibaba runs and projects those lessons back onto every family.

### Per-Run Scores

For eval logs, the combined score is

$$
\mathrm{combined} = \mathrm{MMD}^2 + 0.2 (1 - \mathrm{recall})
$$

where lower is better.

For train logs, the script parses the logged EMA values

$$
\mathrm{train\_combined}_e = \mathrm{EMA\_MMD}^2_e + 0.2 (1 - \mathrm{EMA\_recall}_e)
$$

for each logged epoch \( e \), then keeps

$$
\mathrm{best\_train\_combined} = \min_e \mathrm{train\_combined}_e
$$

The best evaluated checkpoint for a corpus is the run with minimum `combined`.  
The best evaluated recall checkpoint is the run with maximum recall.  
The frontier train-only checkpoint is the run with minimum `best_train_combined`.

### Feature-On Minus Feature-Off Deltas

For each corpus and each switchable training feature such as PCF, multi-scale critic, mixed-type recovery, retrieval memory, or block sampling, R computes

$$
\Delta = \overline{s}_{\mathrm{on}} - \overline{s}_{\mathrm{off}}
$$

where \( s \) is either `combined` for evaluated runs or `best_train_combined` for train-history runs.

For example, the eval-time PCF effect is

$$
\Delta_{\mathrm{pcf, eval}} =
\frac{1}{n_{\mathrm{on}}} \sum_{i : \mathrm{pcf}_i = 1} \mathrm{combined}_i
-
\frac{1}{n_{\mathrm{off}}} \sum_{i : \mathrm{pcf}_i = 0} \mathrm{combined}_i
$$

Negative values mean the feature helped because lower combined score is better.

### Status Labels

Each feature status is assigned deterministically from these deltas.

`validated` means:

$$
\Delta_{\mathrm{eval}} < 0 \quad \text{and} \quad n_{\mathrm{eval,on}} \geq 3
$$

`promising` means the eval evidence above is absent but:

$$
\Delta_{\mathrm{train}} < 0 \quad \text{and} \quad n_{\mathrm{train,on}} \geq 2
$$

`mixed` means some finite evidence exists but the sign is not favorable.  
`unknown` means there is not enough finite evidence.  
`not-primary` is forced for methods that are not a good fit for the family type:

- structured-table families: PCF, multi-scale critic, mixed-type recovery, retrieval memory
- aggregate-time-series families: mixed-type recovery, retrieval memory

### Anchor Distance

Each family is mapped to the closest learned anchor:

- `alibaba__alibaba`
- `s3-cache-datasets__tencentBlock` if present, otherwise `s3-cache-datasets__2020_tencentBlock`

The feature-space distance uses a robustly scaled root-mean-square gap over

- `heterogeneity_score`
- `suggested_modes`
- `write_ratio`
- `reuse_ratio`
- `burstiness_cv`
- `iat_q50`
- `obj_size_q50`
- `tenant_unique`
- `hurst`
- `block_random_distance_ratio`

For feature \( j \), let \( r_j \) be the median absolute deviation over all families:

$$
r_j = \mathrm{median}_i \left| x_{ij} - \mathrm{median}_k x_{kj} \right|
$$

If \( r_j \) is zero or unavailable, the script falls back to the sample standard deviation for that feature.

For a family \( f \) and anchor \( a \), the anchor distance is

$$
d(f, a) =
\sqrt{
\frac{1}{|J^\ast|}
\sum_{j \in J^\ast}
\left( \frac{x_{fj} - x_{aj}}{r_j} \right)^2
}
$$

where \( J^\ast \) is the set of features with finite values for both the family and the anchor and a nonzero scale.

The chosen anchor is

$$
a^\ast(f) = \arg \min_a d(f, a)
$$

### Family-Level Recommendation Rules

R derives four binary workload traits:

Persistent ordering:

$$
\mathrm{persistent} =
\mathbf{1}\{
\mathrm{block\_random\_distance\_ratio} < 0.85
\text{ or }
\mathrm{hurst} \geq 0.75
\}
$$

Multimodal family:

$$
\mathrm{multimodal} =
\mathbf{1}\{
\mathrm{suggested\_modes} \geq 4
\text{ or }
\mathrm{heterogeneity\_score} \geq 1.5
\}
$$

Bursty family:

$$
\mathrm{bursty} =
\mathbf{1}\{
\mathrm{burstiness\_cv} \geq 5
\}
$$

Locality-sensitive family:

$$
\mathrm{locality\_sensitive} =
\mathbf{1}\{
\mathrm{reuse\_ratio} \geq 0.2
\}
$$

Sampling advice is then

$$
\mathrm{sampling\_recommendation} =
\begin{cases}
\text{split-by-format-first}, & \text{if } \mathrm{split\_by\_format} = 1 \\
\text{block}, & \text{if } \mathrm{persistent} = 1 \\
\text{random-ok}, & \text{otherwise}
\end{cases}
$$

Regime advice is

$$
\mathrm{regime\_recommendation} =
\begin{cases}
K \approx 8, & \text{if } \mathrm{multimodal} = 1 \text{ and } \mathrm{suggested\_modes} \geq 6 \\
K \approx 4, & \text{if } \mathrm{multimodal} = 1 \text{ and } \mathrm{suggested\_modes} < 6 \\
\text{single}, & \text{otherwise}
\end{cases}
$$

Char-file conditioning is recommended exactly when

$$
\mathrm{family\_kind} \in
\{
\text{request\_sequence},
\text{aggregate\_time\_series}
\}
$$

Candidate conditioning additions are only surfaced for `request_sequence` families, and they come directly from the earlier conditioning audit.

### Model-Aware Outputs

This layer writes:

- [characterizations/MODEL-LEARNINGS.md](/Users/darrell/Zarathustra/characterizations/MODEL-LEARNINGS.md)
- [characterizations/FAMILY-MODEL-GUIDANCE.md](/Users/darrell/Zarathustra/characterizations/FAMILY-MODEL-GUIDANCE.md)
- [characterizations/model_train_runs.csv](/Users/darrell/Zarathustra/characterizations/model_train_runs.csv)
- [characterizations/model_eval_runs.csv](/Users/darrell/Zarathustra/characterizations/model_eval_runs.csv)
- [characterizations/model_corpus_summary.csv](/Users/darrell/Zarathustra/characterizations/model_corpus_summary.csv)
- [characterizations/family_model_guidance.csv](/Users/darrell/Zarathustra/characterizations/family_model_guidance.csv)

It also appends `## Model-Aware Guidance` to each family report under [characterizations/families](/Users/darrell/Zarathustra/characterizations/families).

## Higher-Order Moment Pass

On 2026-04-18, R was run on `vinge.local` against the `/tiamat` model-aware feature table:

- input: `/tiamat/zarathustra/r-output/model_aware_20260416_0929/results/all_features.csv`
- output: `/tiamat/zarathustra/r-output/higher_moments_20260418_1058/`

This pass computed standardized central moments through order 6 for every numeric feature within each logical family. For feature \( j \), finite values \( x_1,\dots,x_m \), mean \( \overline{x} \), and sample standard deviation \( s \), the additional moment of order \( r \in \{5,6\} \) is:

$$
\mu_r^{\ast}(j) =
\begin{cases}
\dfrac{1}{m} \sum_{i=1}^{m} \left(\dfrac{x_i - \overline{x}}{s}\right)^r, & m \ge r \text{ and } s > 0 \\
\mathrm{NA}, & \text{otherwise}
\end{cases}
$$

Kurtosis in this pass was also recorded as the raw standardized 4th central moment for comparability with orders 5 and 6.

### Generator-Relevant Tail Results

The block-trace families show extreme positive 5th and 6th moments on inter-arrival and stride surfaces. These are not small corrections to skew/kurtosis; they indicate rare-event tail structure that a mean/variance-driven or ordinary moment-matching recipe can easily miss.

| Family | Metric | n | Skew / M3 | Kurtosis / M4 | M5 | M6 |
|---|---|---:|---:|---:|---:|---:|
| s3-cache-datasets__2020_alibabaBlock | iat_q90 | 1000 | 29.607 | 905.362 | 27861.885 | 858620.510 |
| s3-cache-datasets__2020_alibabaBlock | iat_mean | 1000 | 28.975 | 877.655 | 26793.475 | 819243.700 |
| s3-cache-datasets__2020_alibabaBlock | iat_std | 1000 | 24.889 | 695.715 | 19904.685 | 572523.020 |
| s3-cache-datasets__2020_alibabaBlock | reuse_ratio | 1000 | 18.294 | 389.935 | 8634.077 | 194914.630 |
| s3-cache-datasets__2020_tencentBlock | iat_q50 | 4993 | 37.930 | 1659.769 | 77661.370 | 3756862.900 |
| s3-cache-datasets__2020_tencentBlock | iat_mean | 4993 | 34.486 | 1453.885 | 65836.701 | 3081617.000 |
| s3-cache-datasets__2020_tencentBlock | abs_stride_q50 | 4993 | 27.581 | 1082.962 | 47391.454 | 2156810.200 |
| s3-cache-datasets__2020_tencentBlock | abs_stride_q90 | 4993 | 19.428 | 578.088 | 18906.740 | 636516.900 |

### Modeling Implications

- Higher-order tail shape is concentrated in `iat_*`, `abs_stride_*`, and `reuse_ratio`, not uniformly across every feature.
- Tencent has especially extreme 6th moments in median IAT and median absolute stride, which supports treating rare timing/seek regimes as a structural modeling problem rather than as one more scalar-loss weight.
- Alibaba also has massive IAT and reuse tails, but its current recipe reproduces better than Tencent. That suggests the next Alibaba work should not be generic seed churn; it should test whether the chosen checkpoint captures the rare-event tail metrics.
- A direct high-order-moment loss would probably be numerically brittle. The safer interpretation is diagnostic and architectural: use these moments to identify tail-regime families and then model those regimes explicitly.

### Full-Corpus Leaderboard

The pass was not a sample of a few traces. It consumed the full `all_features.csv` table from the model-aware run and produced:

- 560 finite higher-moment rows
- 22 logical families with enough finite observations for standardized 6th moments
- 307 generator-surface rows across `iat_*`, `abs_stride_*`, `reuse_ratio`, object-size, object-popularity, and write-ratio features

The full output files are:

- `/tiamat/zarathustra/r-output/higher_moments_20260418_1058/family_higher_moments.csv`
- `/tiamat/zarathustra/r-output/higher_moments_20260418_1058/family_max_m6_leaderboard.csv`
- `/tiamat/zarathustra/r-output/higher_moments_20260418_1058/all_family_generator_surface_higher_moments.csv`

Top generator-surface M6 rows across all sufficient-size families:

| Family | Metric | n | Skew / M3 | Kurtosis / M4 | M5 | M6 |
|---|---|---:|---:|---:|---:|---:|
| s3-cache-datasets__tencentBlock | abs_stride_q50 | 4991 | 56.516 | 3470.502 | 219447.201 | 14013411.300 |
| s3-cache-datasets__tencentBlock | iat_q50 | 4991 | 54.115 | 3156.205 | 190405.700 | 11666434.600 |
| s3-cache-datasets__2020_tencentBlock | iat_q50 | 4993 | 37.930 | 1659.769 | 77661.370 | 3756862.900 |
| s3-cache-datasets__2020_tencentBlock | iat_mean | 4993 | 34.486 | 1453.885 | 65836.701 | 3081617.000 |
| s3-cache-datasets__2020_tencentBlock | abs_stride_q50 | 4993 | 27.581 | 1082.962 | 47391.454 | 2156810.200 |
| s3-cache-datasets__alibaba | iat_q90 | 1000 | 30.167 | 932.595 | 28947.120 | 899161.400 |
| s3-cache-datasets__alibaba | iat_mean | 1000 | 29.913 | 922.413 | 28559.050 | 884769.500 |
| s3-cache-datasets__2020_alibabaBlock | iat_q90 | 1000 | 29.607 | 905.362 | 27861.885 | 858620.510 |
| alibaba__alibaba | iat_q90 | 999 | 29.592 | 904.453 | 27819.963 | 856898.980 |
| s3-cache-datasets__2020_alibabaBlock | iat_mean | 1000 | 28.975 | 877.655 | 26793.475 | 819243.700 |

## Resource Policy On `vinge.local`

During remote execution:

- jobs ran under `nice -n 10`
- jobs ran under `ionice -c3`
- BLAS/OpenMP thread env vars were pinned to 1
- if `train.py` was active, the R orchestration script set `worker_cap = 6`
- otherwise it would have allowed `worker_cap = 12`

The rebuttal follow-through rerun used the already-normalized corpus and wrote remote heavy output to:

- `/tiamat/zarathustra/r-output/rebuttal_pass_20260405_2033/`

Repo summaries were synced back into:

- [characterizations/README.md](/Users/darrell/Zarathustra/characterizations/README.md)
- [characterizations/families.csv](/Users/darrell/Zarathustra/characterizations/families.csv)
- [characterizations/rollup.json](/Users/darrell/Zarathustra/characterizations/rollup.json)
- [characterizations/families](/Users/darrell/Zarathustra/characterizations/families)

## Limitations

- R never re-opened raw binaries, so all math is downstream of the parser-derived profile surface.
- Structured-table families still get more schema math than request-dynamics math.
- `suggested_modes` is a heuristic, not an ablation-validated optimum.
- `tsfeatures` metrics are package outputs; they were not reimplemented symbolically in this repo.
- The analysis is still file-level. It informs GAN work, but it is not yet a window-level characterization of the actual training sequences.
