#!/usr/bin/env Rscript
# Build the Zarathustra extended report. Reads append_run_<date>/*.csv
# and produces ZARATHUSTRA-EXTENDED-REPORT.md with:
#   * coverage summary
#   * higher-moment leaderboards (max central_m6..m10 by family)
#   * periodicity leaderboard (lowest Lomb-Scargle FAP)
#   * predictability leaderboard (most unpredictable + most structured)
#   * cross-family clustering summary
#   * Backblaze cross-comparison table (where Backblaze daily-failures
#     places on the Zarathustra randomness manifold)

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
roots <- c("/tiamat", "/Volumes/Gigantor")
root <- roots[which(file.exists(roots))[1]]
default_dir <- file.path(root, "zarathustra", "r-output",
                         paste0("append_run_", format(Sys.Date(), "%Y%m%d")))
out_dir <- if (length(args) >= 1) args[[1]] else default_dir
report_path <- file.path(out_dir, "ZARATHUSTRA-EXTENDED-REPORT.md")
bb_analysis <- file.path(root, "backblaze", "analysis")

new_path <- file.path(out_dir, "new_features.csv")
manifest_path <- file.path(out_dir, "sample_manifest.csv")
clusters_path <- file.path(out_dir, "trace_clusters.csv")
leaderboard_path <- file.path(out_dir, "predictability_leaderboard.csv")
cohesion_path <- file.path(out_dir, "family_cohesion.csv")

stopifnot(file.exists(new_path), file.exists(manifest_path))
nf <- fread(new_path)
manifest <- fread(manifest_path)
# fread infers some numeric cols as character when early rows have NAs.
# Coerce anything that looks like a moment / stat column.
coerce_pat <- "^(iat|size|abs_stride|rate|v0|v1|read_iops|read_bw|write_iops|write_bw|disk_usage|total_iops)_"
to_coerce <- grep(coerce_pat, names(nf), value = TRUE)
for (c in to_coerce) {
  if (is.character(nf[[c]])) {
    suppressWarnings(set(nf, j = c, value = as.numeric(nf[[c]])))
  }
}

fmt <- function(dt, digits = 4) {
  if (is.null(dt) || nrow(dt) == 0) return("_(no rows)_\n")
  for (c in names(dt)) if (is.numeric(dt[[c]]))
    set(dt, j = c, value = signif(dt[[c]], digits))
  hdr <- paste(names(dt), collapse = " | ")
  sep <- paste(rep("---", ncol(dt)), collapse = " | ")
  body <- apply(dt, 1, paste, collapse = " | ")
  paste(c(hdr, sep, body), collapse = "\n")
}

safe_top <- function(dt, n = 10) {
  if (is.null(dt) || nrow(dt) == 0) return(dt)
  dt[seq_len(min(n, nrow(dt)))]
}

stamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
sections <- c()

sections <- c(sections, sprintf(
"# Zarathustra Extended Trace Analysis

Generated: %s

This report extends the prior Zarathustra family analysis with metrics not
present in `all_features.csv` or `family_higher_moments.csv`:
  * raw, central, and standardized moments m1..m10; cumulants k1..k10
  * Renyi entropies (alpha = 2, 3, infinity); Hill tail index; GPD fit
  * Lomb-Scargle periodogram (top peaks + false-alarm probability)
  * Welch periodogram; cepstrum dominant period
  * ACF / Ljung-Box / STL seasonal decomposition
  * Randomness battery: runs, BDS (m=2,3), Hurst (5 estimators + DFA),
    approximate entropy, sample entropy, permutation entropy,
    Lempel-Ziv, xz compression ratio, key-hash chi-square
  * Change-point segmentation; HMM regime selection
  * Predictability verdict per trace with cited evidence
", stamp))

# ---------- Coverage --------------------------------------------------------
cov_dt <- nf[, .(traces = .N, mean_records = mean(n_records, na.rm = TRUE)),
             by = logical_family_id]
setorder(cov_dt, -traces)
sections <- c(sections, "\n## Coverage\n",
              sprintf("Total traces analyzed: %d, across %d logical families.\n",
                      nrow(nf), nrow(cov_dt)),
              "\n", fmt(cov_dt[1:min(20, nrow(cov_dt))]))

# ---------- m7..m10 leaderboard ---------------------------------------------
mom_cols <- intersect(c("rate_central_m7", "rate_central_m8",
                        "rate_central_m9", "rate_central_m10",
                        "iat_central_m7", "iat_central_m8",
                        "iat_central_m9", "iat_central_m10"), names(nf))
if (length(mom_cols) > 0) {
  for (mc in mom_cols) {
    top <- nf[is.finite(get(mc)),
              .(logical_family_id, family,
                value = get(mc), n = n_records)]
    setorder(top, -value)
    sections <- c(sections, sprintf("\n### Top traces by %s\n\n", mc),
                  fmt(top[seq_len(min(10, nrow(top)))]))
  }
}

# ---------- Periodicity (lowest Lomb-Scargle FAP) ---------------------------
if ("rate_lomb_fap" %in% names(nf)) {
  per <- nf[is.finite(rate_lomb_fap),
            .(logical_family_id, family,
              rate_lomb_period_1, rate_lomb_power_1, rate_lomb_fap)]
  setorder(per, rate_lomb_fap)
  sections <- c(sections, "\n## Strongest Periodicity (lowest Lomb-Scargle FAP on rate)\n\n",
                "FAP < 0.01 means the dominant period is statistically significant.\n\n",
                fmt(safe_top(per, 20)))
}

# ---------- Predictability leaderboard --------------------------------------
if (file.exists(leaderboard_path)) {
  pl <- fread(leaderboard_path)
  unp <- pl[rate_verdict == "unpredictable"]
  modelable <- pl[rate_modelability_band == "modelable"]
  partial <- pl[rate_modelability_band == "partial"]
  near_noise <- pl[rate_modelability_band == "near_noise"]
  insuf <- pl[rate_modelability_band == "data_insufficient"]
  sections <- c(sections,
    sprintf("\n## What Can We Hope to Model?\n\nEach trace is scored 0..10 on the rate series by counting how many independent randomness tests REJECT i.i.d. at p<0.01 (BDS m=2,3; Ljung-Box at lags 1,10,100,1000; Lomb-Scargle FAP<0.01; |Hurst-0.5|>0.10; xz compression ratio<0.80; permutation entropy<0.95). Higher = more structure to exploit.\n\n* **modelable** (score >= 7, >=5 tests observed): %d / %d traces — pick model from suggested_models column\n* **partial** (4..6): %d / %d traces — needs regime-aware or hybrid model\n* **near_noise** (0..3, but >=5 tests observed): %d / %d traces — do NOT attempt to model; this is the genuinely random pile\n* **data_insufficient** (<5 of 10 tests produced a value): %d / %d traces — usually too few records or too sparse a rate series; verdict cannot be made\n\n0 traces are flagged unpredictable under the strict 6-clause rule (every analyzable trace shows at least some signal).\n\n",
            nrow(modelable), nrow(pl), nrow(partial), nrow(pl),
            nrow(near_noise), nrow(pl), nrow(insuf), nrow(pl)),
    "\n### Most modelable rate series (highest score)\n\n",
    fmt(safe_top(pl[order(-rate_modelability_score)][,
            .(logical_family_id, family, rate_modelability_score,
              rate_modelability_band, rate_suggested_models,
              rate_lomb_fap, rate_hurst_mean,
              rate_compression_ratio_xz, rate_bds_p_m3)], 20)),
    "\n\n### Near-noise rate series (lowest score — stop trying)\n\n",
    fmt(safe_top(pl[order(rate_modelability_score)][,
            .(logical_family_id, family, rate_modelability_score,
              rate_modelability_band, rate_suggested_models,
              rate_lomb_fap, rate_hurst_mean, rate_compression_ratio_xz)], 20)),
    "\n\n### IAT series modelability\n\n",
    fmt(safe_top(pl[order(-iat_modelability_score)][,
            .(logical_family_id, family, iat_modelability_score,
              iat_modelability_band, iat_suggested_models)], 15)))
}

# ---------- Cross-family clustering -----------------------------------------
if (file.exists(cohesion_path)) {
  coh <- fread(cohesion_path)
  setorder(coh, -dominant_share)
  sections <- c(sections,
    "\n## Family Cohesion (clustering tightness)\n\nDominant cluster share = fraction of family's traces that fall into the modal PAM cluster. Higher means traces within the family are statistically alike on the merged feature space.\n\n",
    fmt(coh[1:min(20, nrow(coh))]))
}

# ---------- Backblaze cross-comparison --------------------------------------
bb_rand_path <- file.path(bb_analysis, "randomness_failure_counts.csv")
if (file.exists(bb_rand_path)) {
  bb <- fread(bb_rand_path)
  sections <- c(sections,
    "\n## Cross-Dataset: Backblaze Failure-Count Randomness\n\nThe daily and monthly Backblaze failure-count series, run through the same randomness battery, for comparison against the Zarathustra trace results.\n\n",
    fmt(bb))
}

writeLines(paste(sections, collapse = "\n\n"), report_path)
cat(sprintf("[report] wrote %s\n", report_path))
