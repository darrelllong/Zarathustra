#!/usr/bin/env Rscript
# Apply the same 0..10 modelability score to the Backblaze daily and
# monthly failure-count series, mirroring predictability_verdict.R.

suppressPackageStartupMessages({ library(data.table); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
roots <- c("/tiamat", "/Volumes/Gigantor")
root <- roots[which(file.exists(roots))[1]]
bb_analysis <- if (length(args) >= 1) args[[1]] else file.path(root, "backblaze", "analysis")
report_path <- if (length(args) >= 2) args[[2]] else file.path(root, "backblaze", "REPORT.md")

rand <- fread(file.path(bb_analysis, "randomness_failure_counts.csv"))
peri <- fread(file.path(bb_analysis, "periodicity_failure_counts.csv"))
spec <- fread(file.path(bb_analysis, "spectral_failure_counts.csv"))

score_one <- function(this_series) {
  r <- rand[series == this_series]
  if (nrow(r) == 0) return(NULL)
  bds_m2_p <- r$bds_m2_p[1]; bds_m3_p <- r$bds_m3_p[1]
  hu <- mean(c(r$hurst_simple[1], r$hurst_rs[1], r$hurst_corrected_emp[1],
               r$hurst_aggvar[1]), na.rm = TRUE)
  cr <- r$compression_ratio_xz[1]; pe <- r$permutation_entropy_norm[1]
  lomb_fap <- min(spec[series == this_series & method == "lomb"]$p_value, na.rm = TRUE)
  if (!is.finite(lomb_fap)) lomb_fap <- NA
  lb_lags <- peri[test == "ljung_box" & series == this_series]
  lb1 <- lb_lags[lag == 1]$p_value; if (!length(lb1)) lb1 <- NA
  lb10 <- lb_lags[lag <= 10][.N]$p_value; if (!length(lb10)) lb10 <- NA
  lb100 <- lb_lags[lag <= 100][.N]$p_value; if (!length(lb100)) lb100 <- NA
  lb1000 <- lb_lags[lag <= 1000][.N]$p_value; if (!length(lb1000)) lb1000 <- NA

  s_bds_m2  <- isTRUE(is.finite(bds_m2_p)  && bds_m2_p  < 0.01)
  s_bds_m3  <- isTRUE(is.finite(bds_m3_p)  && bds_m3_p  < 0.01)
  s_lb_1    <- isTRUE(is.finite(lb1)       && lb1       < 0.01)
  s_lb_10   <- isTRUE(is.finite(lb10)      && lb10      < 0.01)
  s_lb_100  <- isTRUE(is.finite(lb100)     && lb100     < 0.01)
  s_lb_1000 <- isTRUE(is.finite(lb1000)    && lb1000    < 0.01)
  s_lomb    <- isTRUE(is.finite(lomb_fap)  && lomb_fap  < 0.01)
  s_hurst   <- isTRUE(is.finite(hu) && abs(hu - 0.5) > 0.10)
  s_compr   <- isTRUE(is.finite(cr) && cr < 0.80)
  s_perm    <- isTRUE(is.finite(pe) && pe < 0.95)
  score <- sum(s_bds_m2, s_bds_m3, s_lb_1, s_lb_10, s_lb_100, s_lb_1000,
               s_lomb, s_hurst, s_compr, s_perm)
  observed_tests <- sum(is.finite(c(bds_m2_p, bds_m3_p, lb1, lb10, lb100,
                                    lb1000, lomb_fap, hu, cr, pe)))
  band <- if (observed_tests < 5) "data_insufficient"
          else if (score >= 7) "modelable"
          else if (score >= 4) "partial"
          else "near_noise"
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

  data.table(
    series = this_series, score = score, band = band,
    observed_tests = observed_tests,
    suggested_models = paste(unique(hints), collapse = ";"),
    bds_m2_p = bds_m2_p, bds_m3_p = bds_m3_p,
    ljung_box_p_lag1 = lb1, ljung_box_p_lag10 = lb10,
    ljung_box_p_lag100 = lb100, ljung_box_p_lag1000 = lb1000,
    lomb_fap = lomb_fap, hurst_mean = hu,
    compression_ratio_xz = cr, permutation_entropy = pe
  )
}

result <- rbindlist(lapply(c("daily", "monthly"), score_one), fill = TRUE)
fwrite(result, file.path(bb_analysis, "modelability_failure_counts.csv"))

# Append section to REPORT.md
fmt_dt <- function(dt) {
  for (c in names(dt)) if (is.numeric(dt[[c]]))
    set(dt, j = c, value = signif(dt[[c]], 4))
  hdr <- paste(names(dt), collapse = " | ")
  sep <- paste(rep("---", ncol(dt)), collapse = " | ")
  body <- apply(dt, 1, paste, collapse = " | ")
  paste(c(hdr, sep, body), collapse = "\n")
}

stamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
section <- c(
  sprintf("\n\n## Modelability Score (Appended %s)\n\n", stamp),
  "Same 0..10 score and bands used for the Zarathustra trace corpus,\n",
  "applied to the Backblaze daily and monthly failure-count series:\n\n",
  fmt_dt(result), "\n"
)
cat(paste(section, collapse = ""), file = report_path, append = TRUE)

cat(sprintf("[bb-modelability] daily=%s/10 (%s)  monthly=%s/10 (%s)\n",
            result$score[1], result$band[1],
            result$score[2], result$band[2]))
