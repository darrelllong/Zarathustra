#!/usr/bin/env Rscript
# Fourier / spectral analysis of I/O trace time series
# Analyzes: ts_delta (inter-arrival times), obj_size, stride patterns, reuse
# Per peer review: "Stop agreeing that spectral analysis is important and then not running it."

suppressPackageStartupMessages({
  library(stats)
})

# --- Binary reader for oracle_general format (24 bytes/record) ---
# Fields: ts(u32), obj_id(u64), obj_size(u32), vtime(i32), op(i16), tenant(i16)
read_oracle_general <- function(path) {
  # Handle .zst compression
  if (grepl("\\.zst$", path)) {
    tmp <- tempfile()
    system2("zstd", c("-d", "-q", "-o", tmp, path))
    con <- file(tmp, "rb")
    on.exit({ close(con); unlink(tmp) })
  } else {
    con <- file(path, "rb")
    on.exit(close(con))
  }

  fsize <- file.info(ifelse(grepl("\\.zst$", path), tmp, path))$size
  n_records <- fsize %/% 24L
  if (n_records < 100) return(NULL)

  # Read in chunks for memory efficiency — cap at 500k records
  n_records <- min(n_records, 500000L)

  ts <- integer(n_records)
  obj_id <- double(n_records)
  obj_size <- integer(n_records)
  op <- integer(n_records)

  for (i in seq_len(n_records)) {
    ts[i] <- readBin(con, "integer", n = 1, size = 4, signed = FALSE, endian = "little")
    obj_id[i] <- readBin(con, "double", n = 1, size = 8, endian = "little")  # raw bits
    obj_size[i] <- readBin(con, "integer", n = 1, size = 4, signed = FALSE, endian = "little")
    readBin(con, "integer", n = 1, size = 4, endian = "little")  # skip vtime
    op[i] <- readBin(con, "integer", n = 1, size = 2, signed = TRUE, endian = "little")
    readBin(con, "integer", n = 1, size = 2, endian = "little")  # skip tenant
  }

  data.frame(
    ts = as.numeric(ts),
    obj_id_raw = obj_id,
    obj_size = as.numeric(obj_size),
    op = op
  )
}

# --- Compute derived series ---
derive_series <- function(df) {
  n <- nrow(df)
  # Inter-arrival times (delta-encoded timestamps)
  ts_delta <- diff(df$ts)
  ts_delta[ts_delta < 0] <- 0  # handle wraparound

  # Object size (already have it)
  obj_size <- df$obj_size[2:n]

  # Reuse: binary flag — did we see this obj_id before?
  seen <- new.env(hash = TRUE, parent = emptyenv())
  reuse <- logical(n)
  for (i in seq_len(n)) {
    key <- as.character(df$obj_id_raw[i])
    if (exists(key, envir = seen)) {
      reuse[i] <- TRUE
    } else {
      assign(key, TRUE, envir = seen)
    }
  }
  reuse <- as.integer(reuse[2:n])

  # Stride: signed distance to last occurrence of same obj_id (0 if first access)
  last_pos <- new.env(hash = TRUE, parent = emptyenv())
  stride <- integer(n)
  for (i in seq_len(n)) {
    key <- as.character(df$obj_id_raw[i])
    if (exists(key, envir = last_pos)) {
      stride[i] <- i - get(key, envir = last_pos)
    }
    assign(key, i, envir = last_pos)
  }
  stride <- stride[2:n]

  data.frame(ts_delta = ts_delta, obj_size = obj_size, reuse = reuse, stride = stride)
}

# --- Spectral analysis of a single series ---
analyze_spectrum <- function(x, name, max_len = 100000) {
  x <- x[is.finite(x)]
  if (length(x) < 256) return(NULL)
  if (length(x) > max_len) x <- x[1:max_len]

  # Remove mean, apply log1p for heavy-tailed series
  if (name %in% c("ts_delta", "obj_size", "stride")) {
    x <- log1p(abs(x)) * sign(x)
  }
  x <- x - mean(x)

  # Compute periodogram
  n <- length(x)
  spec <- spec.pgram(x, spans = c(7, 7), taper = 0.1, plot = FALSE, detrend = TRUE)

  # Find dominant frequencies
  psd <- spec$spec
  freq <- spec$freq
  top_idx <- order(psd, decreasing = TRUE)[1:min(10, length(psd))]

  # Compute spectral entropy (0 = pure tone, 1 = white noise)
  psd_norm <- psd / sum(psd)
  entropy <- -sum(psd_norm * log2(psd_norm + 1e-15)) / log2(length(psd_norm))

  # Compute spectral edge frequency (95% of power below this)
  cumpower <- cumsum(psd) / sum(psd)
  edge_95 <- freq[which(cumpower >= 0.95)[1]]
  edge_50 <- freq[which(cumpower >= 0.50)[1]]

  list(
    name = name,
    n_samples = n,
    entropy = entropy,
    edge_50 = edge_50,
    edge_95 = edge_95,
    top_freqs = freq[top_idx],
    top_powers = psd[top_idx] / max(psd),
    top_periods = 1.0 / freq[top_idx],
    freq = freq,
    psd = psd
  )
}

# --- Main ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript fourier_analysis.R <trace_dir> [format] [max_files]\n")
  cat("  trace_dir: directory containing trace files\n")
  cat("  format: oracle_general (default)\n")
  cat("  max_files: max files to analyze (default 20)\n")
  quit(status = 1)
}

trace_dir <- args[1]
fmt <- ifelse(length(args) >= 2, args[2], "oracle_general")
max_files <- ifelse(length(args) >= 3, as.integer(args[3]), 20L)

# Find trace files
files <- list.files(trace_dir, full.names = TRUE, recursive = FALSE)
files <- files[!grepl("README|characterization|\\.json|\\.csv|\\.md", files)]
if (length(files) > max_files) {
  set.seed(42)
  files <- sample(files, max_files)
}

cat(sprintf("=== Fourier / Spectral Analysis ===\n"))
cat(sprintf("Directory: %s\n", trace_dir))
cat(sprintf("Files to analyze: %d\n", length(files)))
cat(sprintf("Format: %s\n\n", fmt))

# Accumulate series across files
all_ts_delta <- numeric(0)
all_obj_size <- numeric(0)
all_reuse <- integer(0)
all_stride <- numeric(0)
n_files_ok <- 0

for (f in files) {
  cat(sprintf("  Reading %s ... ", basename(f)))
  df <- tryCatch(read_oracle_general(f), error = function(e) NULL)
  if (is.null(df) || nrow(df) < 200) {
    cat("skipped (too small)\n")
    next
  }

  series <- derive_series(df)
  all_ts_delta <- c(all_ts_delta, series$ts_delta)
  all_obj_size <- c(all_obj_size, series$obj_size)
  all_reuse <- c(all_reuse, series$reuse)
  all_stride <- c(all_stride, series$stride)
  n_files_ok <- n_files_ok + 1
  cat(sprintf("%d records\n", nrow(df)))
}

cat(sprintf("\nTotal records across %d files: %d\n\n", n_files_ok, length(all_ts_delta)))

# --- Analyze each series ---
series_list <- list(
  list(data = all_ts_delta, name = "ts_delta"),
  list(data = all_obj_size, name = "obj_size"),
  list(data = all_reuse, name = "reuse"),
  list(data = all_stride[all_stride > 0], name = "stride")
)

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("SPECTRAL ANALYSIS RESULTS\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

for (s in series_list) {
  result <- analyze_spectrum(s$data, s$name)
  if (is.null(result)) {
    cat(sprintf("--- %s: insufficient data ---\n\n", s$name))
    next
  }

  cat(sprintf("--- %s (%d samples) ---\n", result$name, result$n_samples))
  cat(sprintf("  Spectral entropy:     %.4f  (0=pure tone, 1=white noise)\n", result$entropy))
  cat(sprintf("  50%% power edge freq:  %.6f  (period = %.1f samples)\n", result$edge_50, 1/result$edge_50))
  cat(sprintf("  95%% power edge freq:  %.6f  (period = %.1f samples)\n", result$edge_95, 1/result$edge_95))
  cat(sprintf("  Top 5 dominant frequencies:\n"))
  for (i in 1:min(5, length(result$top_freqs))) {
    cat(sprintf("    f=%.6f  period=%.1f samples  rel_power=%.3f\n",
                result$top_freqs[i], result$top_periods[i], result$top_powers[i]))
  }

  # Check for significant periodicity
  if (result$entropy < 0.85) {
    cat(sprintf("  ** SIGNIFICANT PERIODICITY DETECTED (entropy < 0.85) **\n"))
  } else if (result$entropy < 0.95) {
    cat(sprintf("  ~ Mild periodicity (entropy 0.85-0.95)\n"))
  } else {
    cat(sprintf("  ~ Approximately white noise (entropy > 0.95)\n"))
  }
  cat("\n")
}

# --- Summary ---
cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("SUMMARY & IMPLICATIONS FOR LLGAN\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

cat("Key questions answered:\n")
cat("1. Are there dominant burst frequencies the LSTM should capture?\n")
cat("2. Is the IAT series closer to white noise or structured periodicity?\n")
cat("3. Does reuse have temporal rhythm (periodic re-access patterns)?\n")
cat("4. What stride patterns exist at the spectral level?\n\n")

cat("If entropy is high (>0.95): series is approximately random — LSTM's\n")
cat("  12-step window may be sufficient. No long-range spectral structure to miss.\n")
cat("If entropy is low (<0.85): strong periodicity exists — LSTM needs\n")
cat("  longer context or explicit frequency-aware mechanism to capture it.\n")
cat("If 50% power edge is at very low frequency: most energy is in slow\n")
cat("  oscillations that span many windows — chunk-continuity or hierarchical\n")
cat("  modeling is critical.\n\n")

cat("Done.\n")
