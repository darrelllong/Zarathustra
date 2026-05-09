#!/usr/bin/env Rscript
# Per-trace predictability verdict with cited evidence.
#
# Decision rule:
#   `unpredictable` iff ALL of the following hold; else `predictable`
#   with the dominant rejecting test reported.
#     1. BDS p (m=2 AND m=3) > 0.05
#     2. Ljung-Box p > 0.05 at lags {1, 10, 100, 1000}
#     3. Lomb-Scargle FAP > 0.01 (no significant period)
#     4. |Hurst - 0.5| <= 0.05 averaged across the available estimators
#     5. compression_ratio_xz >= 0.95
#     6. approx_entropy within bootstrap CI of i.i.d. surrogate
#        (we approximate this clause with: |perm_entropy - 1| <= 0.05,
#         i.e. permutation entropy near max ⇒ near i.i.d.)
#
# Emits:
#   predictability_leaderboard.csv  (one row per trace + verdict)
#   predictability/<logical>__<basename>.json  (full evidence JSON)

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1 && (is.na(a) || a == ""))) b else a

args <- commandArgs(trailingOnly = TRUE)
roots <- c("/tiamat", "/Volumes/Gigantor")
root <- roots[which(file.exists(roots))[1]]
default_dir <- file.path(root, "zarathustra", "r-output",
                         paste0("append_run_", format(Sys.Date(), "%Y%m%d")))
out_dir <- if (length(args) >= 1) args[[1]] else default_dir
new_path <- file.path(out_dir, "new_features.csv")
verdict_dir <- file.path(out_dir, "predictability")
dir.create(verdict_dir, recursive = TRUE, showWarnings = FALSE)

stopifnot(file.exists(new_path))
nf <- fread(new_path)

is_unpredictable <- function(r, prefix = "rate") {
  px <- function(x) paste(prefix, x, sep = "_")
  bds_m2 <- r[[px("bds_p_m2")]]; bds_m3 <- r[[px("bds_p_m3")]]
  lb1 <- r[[px("ljung_box_p_lag1")]]; lb10 <- r[[px("ljung_box_p_lag10")]]
  lb100 <- r[[px("ljung_box_p_lag100")]]; lb1000 <- r[[px("ljung_box_p_lag1000")]]
  lomb_fap <- r[[px("lomb_fap")]]
  hu <- mean(c(r[[px("hurst_simple")]], r[[px("hurst_rs")]],
               r[[px("hurst_emp")]], r[[px("hurst_aggvar")]]),
             na.rm = TRUE)
  cr <- r[[px("compression_ratio_xz")]]
  pe <- r[[px("perm_entropy")]]

  evidence <- list()
  evidence$bds_p_m2 <- bds_m2; evidence$bds_p_m3 <- bds_m3
  evidence$ljung_box_p_lag1 <- lb1; evidence$ljung_box_p_lag10 <- lb10
  evidence$ljung_box_p_lag100 <- lb100; evidence$ljung_box_p_lag1000 <- lb1000
  evidence$lomb_fap <- lomb_fap
  evidence$hurst_mean <- hu
  evidence$compression_ratio_xz <- cr
  evidence$perm_entropy <- pe

  # Unpredictability clauses (fail-to-reject H0 = i.i.d.)
  clauses <- c(
    bds_pass = isTRUE(bds_m2 > 0.05) && isTRUE(bds_m3 > 0.05),
    ljung_box_pass = isTRUE(lb1 > 0.05) && isTRUE(lb10 > 0.05) &&
                     isTRUE(lb100 > 0.05) && isTRUE(lb1000 > 0.05),
    lomb_pass = isTRUE(lomb_fap > 0.01),
    hurst_pass = isTRUE(is.finite(hu) && abs(hu - 0.5) <= 0.05),
    compress_pass = isTRUE(cr >= 0.95),
    perm_ent_pass = isTRUE(is.finite(pe) && abs(pe - 1) <= 0.05)
  )
  failed <- names(clauses)[!clauses]

  # Modelability score: number of structure tests that REJECT i.i.d. at p<0.01
  # (or whose effect-size threshold is exceeded). Out of 10. Higher = more
  # statistically modelable.
  s_bds_m2  <- isTRUE(is.finite(bds_m2)  && bds_m2  < 0.01)
  s_bds_m3  <- isTRUE(is.finite(bds_m3)  && bds_m3  < 0.01)
  s_lb_1    <- isTRUE(is.finite(lb1)     && lb1     < 0.01)
  s_lb_10   <- isTRUE(is.finite(lb10)    && lb10    < 0.01)
  s_lb_100  <- isTRUE(is.finite(lb100)   && lb100   < 0.01)
  s_lb_1000 <- isTRUE(is.finite(lb1000)  && lb1000  < 0.01)
  s_lomb    <- isTRUE(is.finite(lomb_fap) && lomb_fap < 0.01)
  s_hurst   <- isTRUE(is.finite(hu) && abs(hu - 0.5) > 0.10)
  s_compr   <- isTRUE(is.finite(cr) && cr < 0.80)
  s_perm    <- isTRUE(is.finite(pe) && pe < 0.95)
  modelability_score <- sum(s_bds_m2, s_bds_m3, s_lb_1, s_lb_10, s_lb_100,
                            s_lb_1000, s_lomb, s_hurst, s_compr, s_perm)

  # Count how many of the 10 tests actually produced a finite p-value/stat.
  observed_tests <- sum(is.finite(c(bds_m2, bds_m3, lb1, lb10, lb100, lb1000,
                                     lomb_fap, hu, cr, pe)))
  modelability_band <- if (observed_tests < 5) "data_insufficient"
                       else if (modelability_score >= 7) "modelable"
                       else if (modelability_score >= 4) "partial"
                       else "near_noise"

  # Suggest dominant model class from which signals fired strongest.
  hints <- c()
  if (s_lomb) hints <- c(hints, "fourier_seasonal")
  if (s_hurst && is.finite(hu)) {
    if (hu > 0.6) hints <- c(hints, "ARFIMA_long_memory")
    else if (hu < 0.4) hints <- c(hints, "antipersistent")
  }
  if (s_compr) hints <- c(hints, "low_entropy_dictionary")
  if (s_lb_1 || s_lb_10) hints <- c(hints, "AR_short_lag")
  if (s_lb_100 || s_lb_1000) hints <- c(hints, "ARMA_long_lag")
  if (s_bds_m3 && !s_lb_100) hints <- c(hints, "nonlinear_GARCH")
  if (length(hints) == 0) hints <- "none"

  verdict <- if (all(clauses, na.rm = FALSE)) "unpredictable" else "predictable"
  list(verdict = verdict, evidence = evidence,
       clauses_passed = sum(clauses, na.rm = TRUE),
       clauses_total = length(clauses),
       dominant_failure = if (length(failed)) failed[1] else NA_character_,
       modelability_score = modelability_score,
       modelability_band = modelability_band,
       observed_tests = observed_tests,
       suggested_models = paste(unique(hints), collapse = ";"))
}

leaderboard_rows <- list()
for (i in seq_len(nrow(nf))) {
  r <- as.list(nf[i])
  res <- is_unpredictable(r, prefix = "rate")
  res_iat <- is_unpredictable(r, prefix = "iat")
  base <- paste0(gsub("/", "_", r$logical_family_id), "__",
                 tools::file_path_sans_ext(basename(r$path %||% "")))
  json_path <- file.path(verdict_dir, paste0(base, ".json"))
  full <- list(
    path = r$path, dataset = r$dataset, family = r$family,
    logical_family_id = r$logical_family_id, kind = r$kind,
    rate = res, iat = res_iat
  )
  writeLines(jsonlite::toJSON(full, auto_unbox = TRUE, pretty = TRUE,
                              null = "null", na = "null", digits = 10),
             json_path)
  leaderboard_rows[[length(leaderboard_rows) + 1L]] <- data.table(
    path = r$path, dataset = r$dataset, family = r$family,
    logical_family_id = r$logical_family_id, kind = r$kind,
    rate_verdict = res$verdict, iat_verdict = res_iat$verdict,
    rate_modelability_score = res$modelability_score,
    rate_modelability_band = res$modelability_band,
    rate_suggested_models = res$suggested_models,
    iat_modelability_score = res_iat$modelability_score,
    iat_modelability_band = res_iat$modelability_band,
    iat_suggested_models = res_iat$suggested_models,
    rate_clauses_passed = res$clauses_passed,
    rate_clauses_total = res$clauses_total,
    rate_dominant_failure = res$dominant_failure,
    iat_clauses_passed = res_iat$clauses_passed,
    iat_clauses_total = res_iat$clauses_total,
    iat_dominant_failure = res_iat$dominant_failure,
    rate_bds_p_m3 = res$evidence$bds_p_m3,
    rate_ljung_box_p_lag100 = res$evidence$ljung_box_p_lag100,
    rate_lomb_fap = res$evidence$lomb_fap,
    rate_hurst_mean = res$evidence$hurst_mean,
    rate_compression_ratio_xz = res$evidence$compression_ratio_xz,
    rate_perm_entropy = res$evidence$perm_entropy
  )
}
leaderboard <- rbindlist(leaderboard_rows, use.names = TRUE, fill = TRUE)
setorder(leaderboard, -rate_modelability_score, -iat_modelability_score, dataset, family)
fwrite(leaderboard, file.path(out_dir, "predictability_leaderboard.csv"))

# Per-band summary
band_summary <- leaderboard[, .(
  n = .N,
  median_score = median(rate_modelability_score, na.rm = TRUE)
), by = .(rate_modelability_band)]
fwrite(band_summary, file.path(out_dir, "modelability_band_summary.csv"))

cat(sprintf("[verdict] %d traces  unpredictable: rate=%d iat=%d\n",
            nrow(leaderboard),
            sum(leaderboard$rate_verdict == "unpredictable", na.rm = TRUE),
            sum(leaderboard$iat_verdict == "unpredictable", na.rm = TRUE)))
cat("[verdict] modelability band counts (rate series):\n")
for (b in c("modelable", "partial", "near_noise", "data_insufficient")) {
  n <- sum(leaderboard$rate_modelability_band == b, na.rm = TRUE)
  cat(sprintf("  %-18s  %d / %d\n", b, n, nrow(leaderboard)))
}
