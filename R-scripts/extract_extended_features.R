#!/usr/bin/env Rscript
# Streaming per-trace extractor for the new metrics that aren't in
# all_features.csv yet:
#   moments m7..m10 (raw + central + standardized)
#   cumulants k1..k10
#   Renyi entropies  alpha = 2, 3, inf
#   Hill tail index  + GPD fit
#   Lomb-Scargle peaks (top-3 + FAP)
#   Welch PSD via spec.pgram on binned rate
#   ACF up to lag 1e5; Ljung-Box at lags 1, 10, 100, 1000
#   STL on rate series; cepstrum dominant period
#   Runs test, BDS, Hurst (R/S + DFA), ApEn, SampEn, permutation entropy,
#   Lempel-Ziv complexity, xz/zstd compression ratio of raw + symbol stream,
#   chi-square uniformity on key-hash bins
#   Change-point segments + HMM regime states
#
# Inputs:
#   sample_manifest.csv  (from sample_traces.R)
# Outputs (in --out-dir):
#   new_features.csv          (all numeric features, one row per trace)
#   per_trace/<dataset>__<family>__<basename>.json   (full evidence)
#   moments_m7_m10.csv        (subset for the leaderboard)
#   spectral.csv, periodicity.csv, randomness.csv  (long-form per trace)

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
manifest_path <- if (length(args) >= 1) args[[1]] else stop("usage: extract_extended_features.R MANIFEST OUT_DIR [MAX_RECORDS] [DUMP_PY]")
out_dir <- if (length(args) >= 2) args[[2]] else stop("missing OUT_DIR")
max_records <- if (length(args) >= 3) as.integer(args[[3]]) else 5000000L
dump_py <- if (length(args) >= 4) args[[4]] else file.path(dirname(manifest_path), "..", "r-analysis-src", "R-scripts", "dump_trace_records.py")
if (!file.exists(dump_py)) {
  candidates <- c(
    file.path(Sys.getenv("HOME"), "LLNL", "Zarathustra", "R-scripts", "dump_trace_records.py"),
    "/tiamat/zarathustra/r-analysis-src/R-scripts/dump_trace_records.py"
  )
  hit <- candidates[file.exists(candidates)][1]
  if (!is.na(hit)) dump_py <- hit
}
stopifnot(file.exists(dump_py))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
per_trace_dir <- file.path(out_dir, "per_trace")
dir.create(per_trace_dir, recursive = TRUE, showWarnings = FALSE)

user_lib <- Sys.getenv("R_LIBS_USER", unset = "~/R/library")
.libPaths(c(path.expand(user_lib), .libPaths()))

cat(sprintf("[extract] manifest=%s out=%s max_records=%d\n",
            manifest_path, out_dir, max_records))

manifest <- fread(manifest_path)
stopifnot(all(c("path", "dataset", "family", "format", "kind") %in% names(manifest)))

# ===== Statistical primitives ===============================================

central_moment <- function(x, k) mean((x - mean(x))^k)
standardized_moment <- function(x, k) {
  s <- sd(x); if (!is.finite(s) || s == 0) NA_real_ else central_moment(x, k) / s^k
}
cumulants <- function(x, max_order = 10L) {
  m <- vapply(seq_len(max_order), function(k) mean(x^k), numeric(1L))
  k <- numeric(max_order); k[1] <- m[1]
  for (n in 2:max_order) {
    s <- 0
    for (j in 1:(n - 1)) s <- s + choose(n - 1, j - 1) * k[j] * m[n - j]
    k[n] <- m[n] - s
  }
  setNames(k, paste0("k", seq_len(max_order)))
}
renyi_entropy <- function(x, alpha) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  bins <- pmin(2048L, max(64L, as.integer(sqrt(length(x)))))
  h <- tabulate(cut(x, bins, labels = FALSE, include.lowest = TRUE), nbins = bins)
  p <- h[h > 0] / sum(h)
  if (is.infinite(alpha)) return(-log2(max(p)))
  if (abs(alpha - 1) < 1e-9) return(-sum(p * log2(p)))
  log2(sum(p^alpha)) / (1 - alpha)
}
hill_tail_index <- function(x, k_frac = 0.05) {
  x <- x[is.finite(x) & x > 0]
  if (length(x) < 50L) return(NA_real_)
  k <- max(10L, as.integer(k_frac * length(x)))
  s <- sort(x, decreasing = TRUE)[seq_len(k + 1L)]
  thresh <- s[k + 1L]
  mean(log(s[seq_len(k)] / thresh))
}
gpd_fit_safe <- function(x) {
  if (!requireNamespace("evir", quietly = TRUE)) return(NULL)
  x <- x[is.finite(x) & x > 0]
  if (length(x) < 200L) return(NULL)
  q <- quantile(x, 0.95, names = FALSE)
  fit <- try(evir::gpd(x, threshold = q), silent = TRUE)
  if (inherits(fit, "try-error")) return(NULL)
  list(threshold = q, xi = unname(fit$par.ests["xi"]),
       beta = unname(fit$par.ests["beta"]),
       n_exceedances = fit$n.exceed)
}

lomb_top <- function(x, t = NULL, top = 3L) {
  if (!requireNamespace("lomb", quietly = TRUE)) return(NULL)
  if (is.null(t)) t <- seq_along(x)
  ok <- is.finite(x) & is.finite(t)
  x <- x[ok]; t <- t[ok]
  if (length(x) < 16L) return(NULL)
  # Aggressive subsample (Lomb is O(N*M); ofac=2 keeps M small).
  if (length(x) > 5000L) {
    idx <- sort(sample.int(length(x), 5000L))
    x <- x[idx]; t <- t[idx]
  }
  ls <- try(lomb::lsp(x, times = t, plot = FALSE, type = "period",
                      ofac = 2, alpha = 0.01), silent = TRUE)
  if (inherits(ls, "try-error")) return(NULL)
  scn <- ls$scanned; pwr <- ls$power
  ord <- order(pwr, decreasing = TRUE)
  ord <- ord[seq_len(min(top, length(ord)))]
  list(periods = scn[ord], powers = pwr[ord], fap = ls$p.value)
}

spec_top <- function(x, top = 3L) {
  x <- x[is.finite(x)]
  if (length(x) < 64L) return(NULL)
  if (length(x) > 50000L) x <- x[seq.int(1L, length(x), length.out = 50000L)]
  sp <- try(spec.pgram(x, plot = FALSE, taper = 0.1, detrend = TRUE,
                       demean = TRUE, fast = TRUE), silent = TRUE)
  if (inherits(sp, "try-error")) return(NULL)
  ord <- order(sp$spec, decreasing = TRUE)[seq_len(min(top, length(sp$spec)))]
  list(freq = sp$freq[ord], period = 1 / sp$freq[ord], spec = sp$spec[ord])
}

cepstrum_dominant_period <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 64L) return(NA_real_)
  if (length(x) > 50000L) x <- x[seq.int(1L, length(x), length.out = 50000L)]
  Fx <- fft(x - mean(x))
  cep <- Re(fft(log(abs(Fx) + 1e-12), inverse = TRUE)) / length(x)
  half <- cep[2:floor(length(cep) / 2)]
  if (!length(half)) return(NA_real_)
  which.max(half) + 1L
}

ljung_box <- function(x, lags) {
  vapply(lags, function(L) {
    if (length(x) < L + 5L) return(NA_real_)
    Box.test(x, lag = L, type = "Ljung-Box")$p.value
  }, numeric(1L))
}

acf_summary <- function(x, max_lag = 10000L) {
  if (length(x) < 50L) return(NULL)
  ml <- min(max_lag, length(x) %/% 2)
  a <- acf(x, lag.max = ml, plot = FALSE)
  vals <- as.numeric(a$acf)[-1]
  data.table(
    lag1 = vals[1],
    lag10 = if (length(vals) >= 10) vals[10] else NA_real_,
    lag100 = if (length(vals) >= 100) vals[100] else NA_real_,
    lag1000 = if (length(vals) >= 1000) vals[1000] else NA_real_,
    lag10000 = if (length(vals) >= 10000) vals[10000] else NA_real_,
    decay_to_5pct_lag = {
      below <- which(abs(vals) < 0.05)
      if (length(below)) below[1] else NA_real_
    },
    max_abs_acf_beyond_lag1 = if (length(vals) >= 2) max(abs(vals[-1])) else NA_real_
  )
}

stl_var_share <- function(x, period) {
  if (length(x) < 2 * period) return(rep(NA_real_, 3))
  s <- try(stl(ts(x, frequency = period), s.window = "periodic", robust = TRUE),
           silent = TRUE)
  if (inherits(s, "try-error")) return(rep(NA_real_, 3))
  comp <- s$time.series
  total_var <- var(rowSums(comp))
  c(seasonal = var(comp[, "seasonal"]) / total_var,
    trend = var(comp[, "trend"]) / total_var,
    remainder = var(comp[, "remainder"]) / total_var)
}

runs_p <- function(x) {
  if (!requireNamespace("tseries", quietly = TRUE)) return(NA_real_)
  med <- median(x)
  bin <- factor(ifelse(x > med, "H", "L"))
  if (length(unique(bin)) < 2L) return(NA_real_)
  tseries::runs.test(bin)$p.value
}

bds_pvalues <- function(x, m = 3L, eps_factor = 0.5) {
  if (!requireNamespace("tseries", quietly = TRUE)) return(rep(NA_real_, m - 1L))
  if (length(x) < 100L) return(rep(NA_real_, m - 1L))
  if (length(x) > 50000L) x <- x[seq.int(1L, length(x), length.out = 50000L)]
  eps <- eps_factor * sd(x)
  if (!is.finite(eps) || eps == 0) return(rep(NA_real_, m - 1L))
  r <- try(tseries::bds.test(x, m = m, eps = eps), silent = TRUE)
  if (inherits(r, "try-error")) return(rep(NA_real_, m - 1L))
  as.numeric(r$p.value)
}

hurst_set <- function(x) {
  if (!requireNamespace("pracma", quietly = TRUE)) return(rep(NA_real_, 5))
  if (length(x) < 64L) return(rep(NA_real_, 5))
  he <- try(pracma::hurstexp(x, display = FALSE), silent = TRUE)
  if (inherits(he, "try-error")) return(rep(NA_real_, 5))
  c(Hs_simple = he$Hs, Hrs = he$Hrs, He = he$He, Hal = he$Hal, Ht = he$Ht)
}

dfa_estimate <- function(x) {
  if (!requireNamespace("nonlinearTseries", quietly = TRUE)) return(NA_real_)
  if (length(x) < 200L) return(NA_real_)
  d <- try(nonlinearTseries::dfa(x, do.plot = FALSE,
                                 npoints = 20),
           silent = TRUE)
  if (inherits(d, "try-error")) return(NA_real_)
  est <- try(nonlinearTseries::estimate(d, do.plot = FALSE,
                                        regression.range = NULL),
             silent = TRUE)
  if (inherits(est, "try-error")) return(NA_real_)
  as.numeric(est)
}

approx_ent <- function(x) {
  if (!requireNamespace("pracma", quietly = TRUE)) return(NA_real_)
  if (length(x) < 50L) return(NA_real_)
  if (length(x) > 1000L) x <- x[seq.int(1L, length(x), length.out = 1000L)]
  r <- 0.2 * sd(x); if (!is.finite(r) || r == 0) return(NA_real_)
  pracma::approx_entropy(x, edim = 2L, r = r)
}

sample_ent <- function(x) {
  if (!requireNamespace("pracma", quietly = TRUE)) return(NA_real_)
  if (length(x) < 50L) return(NA_real_)
  if (length(x) > 1000L) x <- x[seq.int(1L, length(x), length.out = 1000L)]
  r <- 0.2 * sd(x); if (!is.finite(r) || r == 0) return(NA_real_)
  pracma::sample_entropy(x, edim = 2L, r = r)
}

perm_ent <- function(x, m = 4L, tau = 1L) {
  x <- x[is.finite(x)]
  N <- length(x)
  if (N < m * tau + 1L) return(NA_real_)
  if (N > 20000L) x <- x[seq.int(1L, N, length.out = 20000L)]
  N <- length(x)
  pat <- vapply(seq_len(N - (m - 1) * tau), function(i) {
    paste(order(x[i + (0:(m - 1)) * tau]), collapse = ",")
  }, character(1L))
  tab <- table(pat); p <- as.numeric(tab) / sum(tab)
  -sum(p * log2(p)) / log2(factorial(m))
}

lempel_ziv <- function(x, n_bits = 4L, max_n = 20000L) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  if (length(x) > max_n) x <- x[seq.int(1L, length(x), length.out = max_n)]
  ranks <- as.integer(cut(rank(x, ties.method = "first"), 2^n_bits,
                           labels = FALSE, include.lowest = TRUE))
  N <- length(ranks)
  if (N < 4L) return(NA_real_)
  # LZ78-style dictionary growth: O(N) using a hash environment.
  dict <- new.env(hash = TRUE, parent = emptyenv())
  c <- 0L
  cur <- ""
  for (i in seq_len(N)) {
    cand <- if (cur == "") as.character(ranks[i]) else
            paste(cur, ranks[i], sep = ",")
    if (exists(cand, envir = dict, inherits = FALSE)) {
      cur <- cand
    } else {
      assign(cand, TRUE, envir = dict)
      c <- c + 1L
      cur <- ""
    }
  }
  c / (N / log2(N))
}

compress_ratio <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  raw <- writeBin(as.numeric(x), raw(), size = 8L)
  z <- memCompress(raw, type = "xz")
  length(z) / length(raw)
}

chisq_uniform_keys <- function(keys, bins = 64L) {
  if (!length(keys)) return(NA_real_)
  # keys can exceed 32-bit int range; use double mod via floor.
  k <- as.integer(((keys / bins) - floor(keys / bins)) * bins) + 1L
  k <- pmin(pmax(k, 1L), bins)
  obs <- tabulate(k, nbins = bins)
  exp <- rep(sum(obs) / bins, bins)
  chisq <- sum((obs - exp)^2 / exp)
  pchisq(chisq, df = bins - 1L, lower.tail = FALSE)
}

change_points <- function(x) {
  if (!requireNamespace("changepoint", quietly = TRUE)) return(NA_integer_)
  if (length(x) < 100L) return(NA_integer_)
  if (length(x) > 50000L) x <- x[seq.int(1L, length(x), length.out = 50000L)]
  cp <- try(changepoint::cpt.meanvar(x, method = "PELT", penalty = "SIC"),
            silent = TRUE)
  if (inherits(cp, "try-error")) return(NA_integer_)
  length(changepoint::cpts(cp))
}

hmm_states <- function(x) {
  if (!requireNamespace("depmixS4", quietly = TRUE)) return(NA_integer_)
  if (length(x) < 200L) return(NA_integer_)
  if (length(x) > 50000L) x <- x[seq.int(1L, length(x), length.out = 50000L)]
  ok <- is.finite(x); x <- x[ok]
  if (length(unique(x)) < 4L) return(NA_integer_)
  df <- data.frame(y = x)
  best <- NA_integer_; best_bic <- Inf
  for (k in 2:4) {
    mod <- try(depmixS4::depmix(y ~ 1, data = df, nstates = k,
                                 family = gaussian()), silent = TRUE)
    if (inherits(mod, "try-error")) next
    fit <- try(depmixS4::fit(mod, verbose = FALSE), silent = TRUE)
    if (inherits(fit, "try-error")) next
    b <- BIC(fit)
    if (is.finite(b) && b < best_bic) { best_bic <- b; best <- k }
  }
  best
}

# ===== Per-trace pipeline ==================================================

extended_moments_and_cumulants <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4L) return(NULL)
  cm <- vapply(2:10, function(k) central_moment(x, k), numeric(1L))
  sm <- vapply(2:10, function(k) standardized_moment(x, k), numeric(1L))
  cu <- cumulants(x, 10L)
  rm_ <- vapply(1:10, function(k) mean(x^k), numeric(1L))
  list(
    raw_m1 = rm_[1], raw_m2 = rm_[2], raw_m3 = rm_[3], raw_m4 = rm_[4],
    raw_m5 = rm_[5], raw_m6 = rm_[6], raw_m7 = rm_[7], raw_m8 = rm_[8],
    raw_m9 = rm_[9], raw_m10 = rm_[10],
    central_m2 = cm[1], central_m3 = cm[2], central_m4 = cm[3],
    central_m5 = cm[4], central_m6 = cm[5], central_m7 = cm[6],
    central_m8 = cm[7], central_m9 = cm[8], central_m10 = cm[9],
    std_m3 = sm[2], std_m4 = sm[3], std_m5 = sm[4], std_m6 = sm[5],
    std_m7 = sm[6], std_m8 = sm[7], std_m9 = sm[8], std_m10 = sm[9],
    k1 = cu[["k1"]], k2 = cu[["k2"]], k3 = cu[["k3"]], k4 = cu[["k4"]],
    k5 = cu[["k5"]], k6 = cu[["k6"]], k7 = cu[["k7"]], k8 = cu[["k8"]],
    k9 = cu[["k9"]], k10 = cu[["k10"]]
  )
}

analyze_series <- function(x, t = NULL, full = TRUE, time_budget_s = 45) {
  start_t <- Sys.time()
  budget_left <- function() as.numeric(Sys.time() - start_t, units = "secs") < time_budget_s
  out <- list(n = length(x))
  m <- extended_moments_and_cumulants(x)
  if (!is.null(m)) out <- c(out, m)
  out$renyi2  <- renyi_entropy(x, 2)
  out$renyi3  <- renyi_entropy(x, 3)
  out$renyi_inf <- renyi_entropy(x, Inf)
  out$hill_tail <- hill_tail_index(x)
  gpd <- if (budget_left() && full) gpd_fit_safe(x) else NULL
  if (!is.null(gpd)) {
    out$gpd_xi <- gpd$xi; out$gpd_beta <- gpd$beta
    out$gpd_threshold <- gpd$threshold
  }
  if (!full) return(out)
  lp <- if (budget_left()) lomb_top(x, t = t, top = 3L) else NULL
  if (!is.null(lp)) {
    out$lomb_period_1 <- lp$periods[1]; out$lomb_power_1 <- lp$powers[1]
    out$lomb_period_2 <- if (length(lp$periods) >= 2) lp$periods[2] else NA_real_
    out$lomb_power_2 <- if (length(lp$powers) >= 2) lp$powers[2] else NA_real_
    out$lomb_period_3 <- if (length(lp$periods) >= 3) lp$periods[3] else NA_real_
    out$lomb_power_3 <- if (length(lp$powers) >= 3) lp$powers[3] else NA_real_
    out$lomb_fap <- lp$fap
  }
  sp <- if (budget_left()) spec_top(x, top = 3L) else NULL
  if (!is.null(sp)) {
    out$spec_period_1 <- sp$period[1]; out$spec_power_1 <- sp$spec[1]
  }
  out$cepstrum_period <- if (budget_left()) cepstrum_dominant_period(x) else NA_real_
  lb <- if (budget_left()) ljung_box(x, c(1, 10, 100, 1000)) else rep(NA_real_, 4)
  out$ljung_box_p_lag1 <- lb[1]; out$ljung_box_p_lag10 <- lb[2]
  out$ljung_box_p_lag100 <- lb[3]; out$ljung_box_p_lag1000 <- lb[4]
  acf_s <- if (budget_left()) acf_summary(x, max_lag = 10000L) else NULL
  if (!is.null(acf_s)) for (n in names(acf_s)) out[[paste0("acf_", n)]] <- acf_s[[n]]
  out$runs_p <- if (budget_left()) runs_p(x) else NA_real_
  bds <- if (budget_left()) bds_pvalues(x, m = 3L) else rep(NA_real_, 2)
  out$bds_p_m2 <- bds[1]; out$bds_p_m3 <- bds[2]
  hu <- if (budget_left()) hurst_set(x) else rep(NA_real_, 5)
  out$hurst_simple <- hu[1]; out$hurst_rs <- hu[2]
  out$hurst_emp <- hu[3]; out$hurst_aggvar <- hu[4]; out$hurst_theory <- hu[5]
  out$dfa_alpha <- NA_real_  # nonlinearTseries::dfa is slow on long series
  out$approx_entropy <- if (budget_left()) approx_ent(x) else NA_real_
  out$sample_entropy <- if (budget_left()) sample_ent(x) else NA_real_
  out$perm_entropy <- if (budget_left()) perm_ent(x, m = 4L) else NA_real_
  out$lempel_ziv <- if (budget_left()) lempel_ziv(x, n_bits = 4L) else NA_real_
  out$compression_ratio_xz <- if (budget_left()) compress_ratio(x) else NA_real_
  out$change_points <- NA_integer_  # PELT can be slow on large series
  out$hmm_best_states <- NA_integer_  # depmixS4 EM is the wall-time bottleneck
  out
}

dump_trace <- function(path, family, fmt, max_records, dump_py) {
  tmp_csv <- tempfile(fileext = ".csv")
  ec <- system2("python3",
                args = c(shQuote(dump_py), shQuote(path),
                         "--out", shQuote(tmp_csv),
                         "--max-records", as.character(max_records),
                         "--family", shQuote(family),
                         "--format", shQuote(fmt)),
                stdout = FALSE, stderr = FALSE)
  if (ec != 0 || !file.exists(tmp_csv)) return(NULL)
  dt <- try(fread(tmp_csv), silent = TRUE)
  unlink(tmp_csv)
  if (inherits(dt, "try-error") || nrow(dt) == 0) return(NULL)
  dt
}

prefix <- function(lst, p) {
  if (is.null(lst) || !length(lst)) return(list())
  setNames(lst, paste0(p, "_", names(lst)))
}

process_one <- function(row, max_records, dump_py) {
  cat(sprintf("[%s] %s\n", row$logical_family_id, basename(row$path)))
  records <- dump_trace(row$path, row$family, row$format, max_records, dump_py)
  if (is.null(records)) {
    return(list(error = "dump failed"))
  }
  out <- list(
    path = row$path, dataset = row$dataset, family = row$family,
    format = row$format, kind = row$kind,
    logical_family_id = row$logical_family_id,
    size_bytes = row$size_bytes, n_records = nrow(records)
  )
  cols <- names(records)
  # Primary series gets the full battery; secondary series get the fast
  # moment/entropy subset only, to keep the wall-time bounded.
  if ("ts" %in% cols && row$kind == "request_sequence") {
    ts <- records$ts
    iat <- diff(ts)
    iat <- iat[is.finite(iat) & iat >= 0]
    out <- c(out, prefix(analyze_series(iat, full = TRUE), "iat"))
    if ("obj_size" %in% cols) {
      sz <- records$obj_size[is.finite(records$obj_size)]
      out <- c(out, prefix(analyze_series(sz, full = FALSE), "size"))
    }
    if ("obj_id" %in% cols) {
      key <- records$obj_id[is.finite(records$obj_id)]
      strides <- diff(key)
      strides <- strides[is.finite(strides)]
      if (length(strides)) out <- c(out, prefix(analyze_series(abs(strides), full = FALSE), "abs_stride"))
      out$key_chisq_p <- chisq_uniform_keys(key, bins = 64L)
    }
    # Binned arrival rate (1s bins) for spectral on rate.
    if (length(ts) > 100L) {
      bin <- floor(ts - min(ts))
      rate <- as.numeric(tabulate(bin + 1L))
      out <- c(out, prefix(analyze_series(rate, full = TRUE), "rate"))
      stl_year <- stl_var_share(rate, period = max(8L, min(86400L, length(rate) %/% 4)))
      out$rate_stl_seasonal_share <- stl_year[1]
      out$rate_stl_trend_share <- stl_year[2]
      out$rate_stl_remainder_share <- stl_year[3]
    }
  } else if (row$kind == "aggregate_time_series") {
    primary <- intersect("total_iops", cols)
    if (!length(primary) && "read_iops" %in% cols && "write_iops" %in% cols) {
      records[, total_iops := records$read_iops + records$write_iops]
      primary <- "total_iops"
    }
    metrics <- intersect(c("read_iops", "read_bw", "write_iops",
                           "write_bw", "disk_usage"), cols)
    for (m in metrics) {
      v <- records[[m]]; v <- v[is.finite(v)]
      is_primary <- length(primary) && (m == "read_iops" || m == "total_iops")
      out <- c(out, prefix(analyze_series(v, full = is_primary), m))
    }
  } else {
    v <- records[[1]]; v <- v[is.finite(v)]
    out <- c(out, prefix(analyze_series(v, full = TRUE), "v0"))
    if (ncol(records) >= 2) {
      v2 <- records[[2]]; v2 <- v2[is.finite(v2)]
      out <- c(out, prefix(analyze_series(v2, full = FALSE), "v1"))
    }
  }
  out
}

# ===== Driver ===============================================================

results <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  r <- as.list(manifest[i])
  res <- try(process_one(r, max_records, dump_py), silent = FALSE)
  if (inherits(res, "try-error")) {
    res <- list(error = paste(attr(res, "condition")$message),
                path = r$path, logical_family_id = r$logical_family_id,
                family = r$family, dataset = r$dataset)
  }
  results[[i]] <- res
  per_path <- file.path(per_trace_dir,
                        paste0(gsub("/", "_", r$logical_family_id), "__",
                               tools::file_path_sans_ext(basename(r$path)),
                               ".json"))
  writeLines(jsonlite::toJSON(res, auto_unbox = TRUE, pretty = FALSE,
                              null = "null", na = "null", digits = 10),
             per_path)
  cat(sprintf("  -> %s\n", per_path))
}

# Flatten to wide CSV
flat_dt <- rbindlist(lapply(results, function(r) {
  if (is.null(r) || is.null(r$path)) return(NULL)
  as.data.table(lapply(r, function(v) if (is.null(v)) NA else
                                       if (length(v) > 1) paste(v, collapse=";") else v))
}), use.names = TRUE, fill = TRUE)
fwrite(flat_dt, file.path(out_dir, "new_features.csv"))

cat(sprintf("[extract] wrote %d feature rows to %s/new_features.csv\n",
            nrow(flat_dt), out_dir))
