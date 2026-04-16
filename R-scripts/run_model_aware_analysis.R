#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required; run install_packages.R first")
}

format_number <- function(x, digits = 3) {
  val <- suppressWarnings(as.numeric(x[[1]] %||% NA_real_))
  if (!is.finite(val)) {
    return("N/A")
  }
  format(round(val, digits), trim = TRUE, scientific = FALSE)
}

safe_num <- function(x) {
  val <- suppressWarnings(as.numeric(x))
  if (length(val) == 0 || !is.finite(val[[1]])) {
    return(NA_real_)
  }
  val[[1]]
}

safe_mean <- function(x) {
  vals <- suppressWarnings(as.numeric(x))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) {
    return(NA_real_)
  }
  mean(vals)
}

detect_family_kind <- function(ml_use_cases) {
  txt <- paste(unique(ml_use_cases[!is.na(ml_use_cases)]), collapse = ",")
  if (grepl("request_sequence", txt, fixed = TRUE)) {
    return("request_sequence")
  }
  if (grepl("aggregate_time_series", txt, fixed = TRUE)) {
    return("aggregate_time_series")
  }
  if (grepl("structured_table", txt, fixed = TRUE)) {
    return("structured_table")
  }
  "unknown"
}

match_first <- function(text, pattern) {
  hit <- regexec(pattern, text, perl = TRUE)
  parts <- regmatches(text, hit)[[1]]
  if (length(parts) < 2) {
    return(NA_character_)
  }
  parts[[2]]
}

flag_present <- function(text, pattern) {
  grepl(pattern, text, perl = TRUE)
}

parse_train_log <- function(path) {
  text <- paste(readLines(path, warn = FALSE), collapse = "\n")
  run <- sub("^train_", "", sub("\\.log$", "", basename(path)))
  corpus <- if (grepl("tencent", run, fixed = TRUE)) "tencent" else if (grepl("alibaba", run, fixed = TRUE)) "alibaba" else "other"

  trace_dir <- match_first(text, "Trace dir:\\s*(\\S+)")
  files_found <- safe_num(match_first(text, "Trace dir:\\s*\\S+\\s*\\((\\d+) files found"))
  sampling_mode <- match_first(text, "Trace dir:\\s*\\S+\\s*\\(\\d+ files found, ([^)]+)\\)")
  if (is.na(sampling_mode)) {
    sampling_mode <- "random"
  }
  regime_sampler <- safe_num(match_first(text, "\\[regime-sampler\\] K=(\\d+)"))
  if (!is.finite(regime_sampler)) {
    regime_sampler <- 0
  }

  epoch_hits <- gregexpr("Epoch\\s+(\\d+)/(\\d+).*?EMA MMD²=([0-9.]+)\\s+recall=([0-9.]+)\\s+comb=([0-9.]+)(\\s+★)?", text, perl = TRUE)
  match_txt <- regmatches(text, epoch_hits)[[1]]
  points <- lapply(match_txt, function(line) {
    parts <- regexec("Epoch\\s+(\\d+)/(\\d+).*?EMA MMD²=([0-9.]+)\\s+recall=([0-9.]+)\\s+comb=([0-9.]+)(\\s+★)?", line, perl = TRUE)
    vals <- regmatches(line, parts)[[1]]
    if (length(vals) < 6) {
      return(NULL)
    }
    data.frame(
      epoch = safe_num(vals[[2]]),
      total_epochs = safe_num(vals[[3]]),
      train_mmd2 = safe_num(vals[[4]]),
      train_recall = safe_num(vals[[5]]),
      train_combined = safe_num(vals[[6]]),
      star = if (length(vals) >= 7) nzchar(vals[[7]]) else FALSE,
      stringsAsFactors = FALSE
    )
  })
  points <- Filter(Negate(is.null), points)
  point_df <- if (length(points) > 0) do.call(rbind, points) else data.frame()

  best_idx <- if (nrow(point_df) > 0) which.min(point_df$train_combined) else integer()
  data.frame(
    run = run,
    corpus = corpus,
    train_log = basename(path),
    trace_dir = trace_dir,
    files_found = files_found,
    sampling_mode = sampling_mode,
    retrieval_memory = flag_present(text, "\\[retrieval-memory\\] enabled"),
    regime_sampler = regime_sampler,
    multi_scale_critic = flag_present(text, "\\[multi-scale critic\\]"),
    mixed_type_recovery = flag_present(text, "\\[mixed-type\\]"),
    pcf = flag_present(text, "\\[PCF\\]"),
    eval_points = nrow(point_df),
    best_train_epoch = if (length(best_idx) == 1) point_df$epoch[[best_idx]] else NA_real_,
    best_train_mmd2 = if (length(best_idx) == 1) point_df$train_mmd2[[best_idx]] else NA_real_,
    best_train_recall = if (length(best_idx) == 1) point_df$train_recall[[best_idx]] else NA_real_,
    best_train_combined = if (length(best_idx) == 1) point_df$train_combined[[best_idx]] else NA_real_,
    last_train_epoch = if (nrow(point_df) > 0) point_df$epoch[[nrow(point_df)]] else NA_real_,
    last_train_combined = if (nrow(point_df) > 0) point_df$train_combined[[nrow(point_df)]] else NA_real_,
    stringsAsFactors = FALSE
  )
}

parse_eval_log <- function(path) {
  text <- paste(readLines(path, warn = FALSE), collapse = "\n")
  checkpoint <- match_first(text, "Checkpoint\\s*:\\s*(\\S+)")
  run <- if (!is.na(checkpoint)) basename(dirname(checkpoint)) else sub("^eval_", "", sub("\\.log$", "", basename(path)))
  corpus <- if (grepl("tencent", run, fixed = TRUE)) "tencent" else if (grepl("alibaba", run, fixed = TRUE)) "alibaba" else "other"

  mmd2 <- safe_num(match_first(text, "\n\\s*MMD²\\s*:\\s*([0-9.]+)"))
  recall <- safe_num(match_first(text, "β-recall\\s*:\\s*([0-9.]+)"))

  data.frame(
    run = run,
    corpus = corpus,
    eval_log = basename(path),
    checkpoint = checkpoint,
    eval_epoch = safe_num(match_first(text, "Epoch\\s*:\\s*(\\d+)")),
    saved_mmd2 = safe_num(match_first(text, "Saved MMD²\\s*:\\s*([0-9.]+)")),
    mmd2 = mmd2,
    precision = safe_num(match_first(text, "α-precision\\s*:\\s*([0-9.]+)")),
    recall = recall,
    density = safe_num(match_first(text, "density\\s*:\\s*([0-9.]+)")),
    coverage = safe_num(match_first(text, "coverage\\s*:\\s*([0-9.]+)")),
    dmdgen = safe_num(match_first(text, "DMD-GEN\\s*:\\s*([0-9.]+)")),
    autocorr = safe_num(match_first(text, "AutoCorr\\s*:\\s*([0-9.]+)")),
    spectral = safe_num(match_first(text, "Spectral\\s*:\\s*([0-9.]+)")),
    context_fid = safe_num(match_first(text, "Context-FID\\s*:\\s*([0-9.]+)")),
    hrc_mae = safe_num(match_first(text, "HRC-MAE\\s*:\\s*([0-9.]+)")),
    real_reuse_rate = safe_num(match_first(text, "reuse rate\\s*:\\s*real=([0-9.]+)")),
    fake_reuse_rate = safe_num(match_first(text, "reuse rate\\s*:\\s*real=[0-9.]+\\s+fake=([0-9.]+)")),
    frozen_bundle = grepl("frozen", basename(path), fixed = TRUE),
    eval_variant = sub("^eval_[^_]+", "", sub("\\.log$", "", basename(path))),
    combined = if (is.finite(mmd2) && is.finite(recall)) mmd2 + 0.2 * (1 - recall) else NA_real_,
    temporal_score = NA_real_,
    stringsAsFactors = FALSE
  )
}

load_rollup_df <- function(path) {
  rollup <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  ids <- names(rollup)
  rows <- lapply(ids, function(id) {
    row <- rollup[[id]]
    data.frame(
      logical_family_id = id,
      dataset = row$dataset %||% NA_character_,
      family = row$family %||% NA_character_,
      files = safe_num(row$files),
      bytes = safe_num(row$bytes),
      formats = paste(row$formats %||% character(), collapse = ","),
      parsers = paste(row$parsers %||% character(), collapse = ","),
      ml_use_cases = paste(row$ml_use_cases %||% character(), collapse = ","),
      heterogeneity_score = safe_num(row$heterogeneity_score),
      suggested_modes = safe_num(row$suggested_modes),
      split_by_format = isTRUE(row$split_by_format),
      kmeans_selected_k = safe_num(row$kmeans_selected_k),
      kmeans_best_k = safe_num(row$kmeans_best_k),
      mclust_components = safe_num(row$mclust_components),
      dbscan_clusters = safe_num(row$dbscan_clusters),
      changepoint_count = safe_num(row$changepoint_count),
      hurst = safe_num(row$hurst),
      block_random_distance_ratio = safe_num(row$block_random_distance_ratio),
      temporal_sampling_recommendation = row$temporal_sampling_recommendation %||% "",
      top_regime_driver = row$top_regime_driver %||% "",
      conditioning_drop_candidates = row$conditioning_drop_candidates %||% "",
      conditioning_add_candidates = row$conditioning_add_candidates %||% "",
      gan_guidance = row$gan_guidance %||% "",
      write_ratio = safe_num(row$write_ratio),
      reuse_ratio = safe_num(row$reuse_ratio),
      burstiness_cv = safe_num(row$burstiness_cv),
      iat_q50 = safe_num(row$iat_q50),
      obj_size_q50 = safe_num(row$obj_size_q50),
      tenant_unique = safe_num(row$tenant_unique),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

mean_diff <- function(df, metric, flag_col, on_value = TRUE, off_value = FALSE) {
  if (!(metric %in% names(df)) || !(flag_col %in% names(df))) {
    return(list(on_mean = NA_real_, off_mean = NA_real_, diff = NA_real_, n_on = 0L, n_off = 0L))
  }
  vals <- df[is.finite(df[[metric]]) & !is.na(df[[flag_col]]), c(metric, flag_col), drop = FALSE]
  if (nrow(vals) == 0) {
    return(list(on_mean = NA_real_, off_mean = NA_real_, diff = NA_real_, n_on = 0L, n_off = 0L))
  }
  on <- vals[vals[[flag_col]] == on_value, metric, drop = TRUE]
  off <- vals[vals[[flag_col]] == off_value, metric, drop = TRUE]
  list(
    on_mean = safe_mean(on),
    off_mean = safe_mean(off),
    diff = safe_mean(on) - safe_mean(off),
    n_on = length(on),
    n_off = length(off)
  )
}

mean_diff_sampling <- function(df, metric) {
  if (!(metric %in% names(df)) || !("sampling_mode" %in% names(df))) {
    return(list(block_mean = NA_real_, random_mean = NA_real_, diff = NA_real_, n_block = 0L, n_random = 0L))
  }
  vals <- df[is.finite(df[[metric]]) & !is.na(df$sampling_mode), c(metric, "sampling_mode"), drop = FALSE]
  if (nrow(vals) == 0) {
    return(list(block_mean = NA_real_, random_mean = NA_real_, diff = NA_real_, n_block = 0L, n_random = 0L))
  }
  block <- vals[vals$sampling_mode == "block sampling", metric, drop = TRUE]
  random <- vals[vals$sampling_mode != "block sampling", metric, drop = TRUE]
  list(
    block_mean = safe_mean(block),
    random_mean = safe_mean(random),
    diff = safe_mean(block) - safe_mean(random),
    n_block = length(block),
    n_random = length(random)
  )
}

get_best_row <- function(df, metric, decreasing = FALSE) {
  vals <- suppressWarnings(as.numeric(df[[metric]]))
  keep <- is.finite(vals)
  if (!any(keep)) {
    return(NULL)
  }
  sub <- df[keep, , drop = FALSE]
  vals <- suppressWarnings(as.numeric(sub[[metric]]))
  idx <- if (decreasing) which.max(vals) else which.min(vals)
  sub[idx, , drop = FALSE]
}

summarize_corpora <- function(train_df, eval_df) {
  corpora <- intersect(c("tencent", "alibaba"), sort(unique(c(train_df$corpus, eval_df$corpus))))
  rows <- lapply(corpora, function(corpus) {
    train_sub <- train_df[train_df$corpus == corpus, , drop = FALSE]
    eval_sub <- eval_df[eval_df$corpus == corpus, , drop = FALSE]
    if (nrow(train_sub) == 0 && nrow(eval_sub) == 0) {
      return(NULL)
    }

    best_eval <- get_best_row(eval_sub, "combined", decreasing = FALSE)
    best_eval_recall <- get_best_row(eval_sub, "recall", decreasing = TRUE)
    best_eval_temporal <- get_best_row(eval_sub, "dmdgen", decreasing = FALSE)
    best_frontier <- get_best_row(train_sub, "best_train_combined", decreasing = FALSE)

    eval_block <- mean_diff_sampling(eval_sub, "combined")
    train_block <- mean_diff_sampling(train_sub, "best_train_combined")
    eval_pcf <- mean_diff(eval_sub, "combined", "pcf")
    train_pcf <- mean_diff(train_sub, "best_train_combined", "pcf")
    eval_multi <- mean_diff(eval_sub, "combined", "multi_scale_critic")
    train_multi <- mean_diff(train_sub, "best_train_combined", "multi_scale_critic")
    eval_mixed <- mean_diff(eval_sub, "combined", "mixed_type_recovery")
    train_mixed <- mean_diff(train_sub, "best_train_combined", "mixed_type_recovery")
    train_retrieval <- mean_diff(train_sub, "best_train_combined", "retrieval_memory")

    data.frame(
      corpus = corpus,
      n_train_runs = nrow(train_sub),
      n_eval_runs = nrow(eval_sub),
      best_eval_run = best_eval$run %||% NA_character_,
      best_eval_combined = safe_num(best_eval$combined),
      best_eval_recall_run = best_eval_recall$run %||% NA_character_,
      best_eval_recall = safe_num(best_eval_recall$recall),
      best_eval_temporal_run = best_eval_temporal$run %||% NA_character_,
      best_eval_dmdgen = safe_num(best_eval_temporal$dmdgen),
      frontier_run = best_frontier$run %||% NA_character_,
      frontier_best_train_combined = safe_num(best_frontier$best_train_combined),
      frontier_best_train_recall = safe_num(best_frontier$best_train_recall),
      eval_block_diff = eval_block$diff,
      eval_block_n_block = eval_block$n_block,
      eval_block_n_random = eval_block$n_random,
      train_block_diff = train_block$diff,
      train_block_n_block = train_block$n_block,
      train_block_n_random = train_block$n_random,
      eval_pcf_diff = eval_pcf$diff,
      eval_pcf_n_on = eval_pcf$n_on,
      eval_pcf_n_off = eval_pcf$n_off,
      train_pcf_diff = train_pcf$diff,
      train_pcf_n_on = train_pcf$n_on,
      train_pcf_n_off = train_pcf$n_off,
      eval_multiscale_diff = eval_multi$diff,
      eval_multiscale_n_on = eval_multi$n_on,
      eval_multiscale_n_off = eval_multi$n_off,
      train_multiscale_diff = train_multi$diff,
      train_multiscale_n_on = train_multi$n_on,
      train_multiscale_n_off = train_multi$n_off,
      eval_mixed_diff = eval_mixed$diff,
      eval_mixed_n_on = eval_mixed$n_on,
      eval_mixed_n_off = eval_mixed$n_off,
      train_mixed_diff = train_mixed$diff,
      train_mixed_n_on = train_mixed$n_on,
      train_mixed_n_off = train_mixed$n_off,
      train_retrieval_diff = train_retrieval$diff,
      train_retrieval_n_on = train_retrieval$n_on,
      train_retrieval_n_off = train_retrieval$n_off,
      stringsAsFactors = FALSE
    )
  })
  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0) {
    data.frame()
  } else {
    do.call(rbind, rows)
  }
}

derive_anchor_profiles <- function(family_df) {
  alibaba_id <- "alibaba__alibaba"
  tencent_id <- if ("s3-cache-datasets__tencentBlock" %in% family_df$logical_family_id) "s3-cache-datasets__tencentBlock" else "s3-cache-datasets__2020_tencentBlock"
  anchors <- family_df[family_df$logical_family_id %in% c(alibaba_id, tencent_id), , drop = FALSE]
  if (!("anchor_name" %in% names(anchors))) {
    anchors$anchor_name <- ifelse(anchors$logical_family_id == alibaba_id, "alibaba", "tencent_block")
  }
  anchors
}

compute_anchor_distance <- function(row, anchor, feature_df, feature_cols) {
  diffs <- numeric()
  for (col in feature_cols) {
    x <- suppressWarnings(as.numeric(row[[col]]))
    y <- suppressWarnings(as.numeric(anchor[[col]]))
    ref <- suppressWarnings(as.numeric(feature_df[[col]]))
    ref <- ref[is.finite(ref)]
    if (!is.finite(x) || !is.finite(y) || length(ref) < 3) {
      next
    }
    scale <- stats::mad(ref, center = stats::median(ref), constant = 1, na.rm = TRUE)
    if (!is.finite(scale) || scale <= 1e-9) {
      scale <- stats::sd(ref, na.rm = TRUE)
    }
    if (!is.finite(scale) || scale <= 1e-9) {
      next
    }
    diffs <- c(diffs, (x - y) / scale)
  }
  if (length(diffs) == 0) {
    return(NA_real_)
  }
  sqrt(mean(diffs ^ 2))
}

status_from_effects <- function(eval_diff, eval_n_on, train_diff, train_n_on, better_when_lower = TRUE) {
  eval_good <- is.finite(eval_diff) && eval_n_on >= 3 && if (better_when_lower) eval_diff < 0 else eval_diff > 0
  train_good <- is.finite(train_diff) && train_n_on >= 2 && if (better_when_lower) train_diff < 0 else train_diff > 0
  if (eval_good) {
    return("validated")
  }
  if (train_good) {
    return("promising")
  }
  if (is.finite(eval_diff) || is.finite(train_diff)) {
    return("mixed")
  }
  "unknown"
}

compose_family_guidance <- function(row, corpus_summary_row, anchor_name) {
  family_kind <- detect_family_kind(row$ml_use_cases)
  persistent <- (is.finite(row$block_random_distance_ratio) && row$block_random_distance_ratio < 0.85) || (is.finite(row$hurst) && row$hurst >= 0.75)
  multimodal <- (is.finite(row$suggested_modes) && row$suggested_modes >= 4) || (is.finite(row$heterogeneity_score) && row$heterogeneity_score >= 1.5)
  bursty <- is.finite(row$burstiness_cv) && row$burstiness_cv >= 5
  locality_sensitive <- is.finite(row$reuse_ratio) && row$reuse_ratio >= 0.2

  block_status <- status_from_effects(corpus_summary_row$eval_block_diff, corpus_summary_row$eval_block_n_block, corpus_summary_row$train_block_diff, corpus_summary_row$train_block_n_block)
  pcf_status <- status_from_effects(corpus_summary_row$eval_pcf_diff, corpus_summary_row$eval_pcf_n_on, corpus_summary_row$train_pcf_diff, corpus_summary_row$train_pcf_n_on)
  multi_status <- status_from_effects(corpus_summary_row$eval_multiscale_diff, corpus_summary_row$eval_multiscale_n_on, corpus_summary_row$train_multiscale_diff, corpus_summary_row$train_multiscale_n_on)
  mixed_status <- status_from_effects(corpus_summary_row$eval_mixed_diff, corpus_summary_row$eval_mixed_n_on, corpus_summary_row$train_mixed_diff, corpus_summary_row$train_mixed_n_on)
  retrieval_status <- status_from_effects(NA_real_, 0L, corpus_summary_row$train_retrieval_diff, corpus_summary_row$train_retrieval_n_on)

  if (family_kind == "structured_table") {
    pcf_status <- "not-primary"
    multi_status <- "not-primary"
    mixed_status <- "not-primary"
    retrieval_status <- "not-primary"
  } else if (family_kind == "aggregate_time_series") {
    mixed_status <- "not-primary"
    retrieval_status <- "not-primary"
  }

  sampling_recommendation <- if (row$split_by_format) {
    "split-by-format-first"
  } else if (persistent) {
    "block"
  } else {
    "random-ok"
  }
  regime_recommendation <- if (multimodal) {
    if (is.finite(row$suggested_modes) && row$suggested_modes >= 6) "K≈8" else "K≈4"
  } else {
    "single"
  }
  char_file_conditioning <- family_kind %in% c("request_sequence", "aggregate_time_series")
  conditioning_candidates <- if (family_kind == "request_sequence") {
    trimws(row$conditioning_add_candidates %||% "")
  } else {
    ""
  }

  defaults <- character()
  candidates <- character()

  if (sampling_recommendation == "split-by-format-first") {
    defaults <- c(defaults, "split formats before training")
  }
  if (sampling_recommendation == "block") {
    defaults <- c(defaults, "use block or sequential file sampling")
  }
  if (char_file_conditioning) {
    defaults <- c(defaults, "use char-file conditioning")
  }
  if (regime_recommendation != "single") {
    defaults <- c(defaults, paste0("use a regime sampler around ", regime_recommendation))
  }
  if (locality_sensitive) {
    defaults <- c(defaults, "treat locality as first-class in the loss/conditioning stack")
  }
  if (bursty && family_kind != "structured_table") {
    defaults <- c(defaults, "keep burst-sensitive temporal objectives on")
  }

  if (family_kind != "structured_table" && pcf_status %in% c("validated", "promising")) {
    candidates <- c(candidates, paste0("PCF loss is ", pcf_status, " on the ", anchor_name, " corpus"))
  }
  if (family_kind != "structured_table" && multi_status %in% c("validated", "promising") && multimodal) {
    candidates <- c(candidates, paste0("multi-scale critic looks ", multi_status, " for higher-mode families"))
  }
  if (family_kind == "request_sequence" && mixed_status %in% c("validated", "promising")) {
    candidates <- c(candidates, paste0("mixed-type recovery is ", mixed_status, " for request-sequence windows"))
  }
  if (anchor_name == "tencent_block" && retrieval_status %in% c("validated", "promising") && multimodal) {
    candidates <- c(candidates, paste0("retrieval memory is ", retrieval_status, " but still frontier-only"))
  }
  if (nzchar(conditioning_candidates)) {
    candidates <- c(candidates, paste0("test extra conditioning features: ", conditioning_candidates))
  }
  if (family_kind == "structured_table") {
    defaults <- c(defaults, "current window GAN is a weaker fit than it is for request-sequence families")
  }

  rationale <- character()
  if (persistent) {
    rationale <- c(rationale, "ordered files show temporal persistence")
  }
  if (multimodal) {
    rationale <- c(rationale, "family looks multi-regime or high-heterogeneity")
  }
  if (locality_sensitive) {
    rationale <- c(rationale, "reuse/locality is not negligible")
  }
  if (bursty) {
    rationale <- c(rationale, "burstiness is materially above the calmer families")
  }
  if (row$split_by_format) {
    rationale <- c(rationale, "formats/parsers are mixed")
  }
  if (length(rationale) == 0) {
    rationale <- "no single pathological axis dominates this family"
  } else {
    rationale <- paste(rationale, collapse = "; ")
  }

  data.frame(
    logical_family_id = row$logical_family_id,
    dataset = row$dataset,
    family = row$family,
    family_kind = family_kind,
    closest_anchor = anchor_name,
    sampling_recommendation = sampling_recommendation,
    regime_recommendation = regime_recommendation,
    char_file_conditioning = char_file_conditioning,
    pcf_status = pcf_status,
    multiscale_status = multi_status,
    mixed_type_status = mixed_status,
    retrieval_status = retrieval_status,
    conditioning_candidates = conditioning_candidates,
    model_defaults = paste(defaults, collapse = "; "),
    model_candidates = paste(candidates, collapse = "; "),
    rationale = rationale,
    stringsAsFactors = FALSE
  )
}

build_family_guidance <- function(family_df, corpus_summary) {
  anchors <- derive_anchor_profiles(family_df)
  feature_cols <- intersect(
    c("heterogeneity_score", "suggested_modes", "write_ratio", "reuse_ratio", "burstiness_cv", "iat_q50", "obj_size_q50", "tenant_unique", "hurst", "block_random_distance_ratio"),
    names(family_df)
  )

  rows <- lapply(seq_len(nrow(family_df)), function(i) {
    row <- family_df[i, , drop = FALSE]
    dists <- vapply(seq_len(nrow(anchors)), function(j) compute_anchor_distance(row, anchors[j, , drop = FALSE], family_df, feature_cols), numeric(1))
    best_anchor <- if (all(!is.finite(dists))) 1L else which.min(replace(dists, !is.finite(dists), Inf))
    anchor_name <- anchors$anchor_name[[best_anchor]]
    corpus_row <- corpus_summary[corpus_summary$corpus == if (anchor_name == "tencent_block") "tencent" else "alibaba", , drop = FALSE]
    guidance <- compose_family_guidance(row, corpus_row[1, , drop = FALSE], anchor_name)
    guidance$anchor_distance <- dists[[best_anchor]]
    guidance
  })
  do.call(rbind, rows)
}

render_model_learnings <- function(corpus_summary, eval_df, train_df, out_path) {
  lines <- c(
    "# Model Learnings",
    "",
    "This report joins the family characterization pass with the full train/eval history from `vinge.local`.",
    "",
    "## Inventory",
    "",
    paste0("- Train logs parsed: ", nrow(train_df)),
    paste0("- Eval logs parsed: ", nrow(eval_df)),
    ""
  )

  for (corpus in corpus_summary$corpus) {
    row <- corpus_summary[corpus_summary$corpus == corpus, , drop = FALSE]
    eval_rows <- eval_df[eval_df$corpus == corpus, , drop = FALSE]
    eval_rows <- eval_rows[order(eval_rows$combined), , drop = FALSE]
    train_rows <- train_df[train_df$corpus == corpus, , drop = FALSE]
    train_rows <- train_rows[order(train_rows$best_train_combined), , drop = FALSE]

    lines <- c(lines, paste0("## ", toupper(substr(corpus, 1, 1)), substr(corpus, 2, nchar(corpus))), "", "| Item | Value |", "|---|---|")
    lines <- c(lines, paste0("| Evaluated best checkpoint | ", row$best_eval_run, " |"))
    lines <- c(lines, paste0("| Evaluated best combined score | ", format_number(row$best_eval_combined), " |"))
    lines <- c(lines, paste0("| Best evaluated recall checkpoint | ", row$best_eval_recall_run, " |"))
    lines <- c(lines, paste0("| Best evaluated recall | ", format_number(row$best_eval_recall), " |"))
    lines <- c(lines, paste0("| Frontier train-only checkpoint | ", row$frontier_run, " |"))
    lines <- c(lines, paste0("| Frontier best train combined | ", format_number(row$frontier_best_train_combined), " |"))

    lines <- c(lines, "", "### Validated Lessons", "")
    lessons <- character()
    if (is.finite(row$eval_pcf_diff)) {
      lessons <- c(lessons, paste0("PCF-on evaluated runs shift combined score by ", format_number(row$eval_pcf_diff), " relative to PCF-off for this corpus."))
    }
    if (is.finite(row$eval_block_diff)) {
      lessons <- c(lessons, paste0("Block sampling shifts evaluated combined score by ", format_number(row$eval_block_diff), " relative to random sampling."))
    }
    if (length(lessons) == 0) {
      lessons <- "No high-confidence validated deltas were available from completed evals alone."
    }
    for (note in lessons) {
      lines <- c(lines, paste0("- ", note))
    }

    lines <- c(lines, "", "### Frontier Lessons", "")
    frontier <- character()
    if (is.finite(row$train_block_diff)) {
      frontier <- c(frontier, paste0("Block sampling shifts best-train combined by ", format_number(row$train_block_diff), " in the running history."))
    }
    if (is.finite(row$train_pcf_diff)) {
      frontier <- c(frontier, paste0("PCF-on shifts best-train combined by ", format_number(row$train_pcf_diff), " in the running history."))
    }
    if (is.finite(row$train_multiscale_diff)) {
      frontier <- c(frontier, paste0("Multi-scale critic shifts best-train combined by ", format_number(row$train_multiscale_diff), " in the running history."))
    }
    if (is.finite(row$train_mixed_diff)) {
      frontier <- c(frontier, paste0("Mixed-type recovery shifts best-train combined by ", format_number(row$train_mixed_diff), " in the running history."))
    }
    if (is.finite(row$train_retrieval_diff)) {
      frontier <- c(frontier, paste0("Retrieval memory shifts best-train combined by ", format_number(row$train_retrieval_diff), " in the running history."))
    }
    if (length(frontier) == 0) {
      frontier <- "No frontier deltas available."
    }
    for (note in frontier) {
      lines <- c(lines, paste0("- ", note))
    }

    lines <- c(lines, "", "### Top Evaluated Checkpoints", "", "| Run | Combined | Recall | MMD² | DMD-GEN | HRC-MAE |", "|---|---:|---:|---:|---:|---:|")
    for (i in seq_len(min(8, nrow(eval_rows)))) {
      r <- eval_rows[i, , drop = FALSE]
      lines <- c(lines, paste0("| ", r$run, " | ", format_number(r$combined), " | ", format_number(r$recall), " | ", format_number(r$mmd2), " | ", format_number(r$dmdgen), " | ", format_number(r$hrc_mae), " |"))
    }

    lines <- c(lines, "", "### Frontier Train Runs", "", "| Run | Best Train Combined | Recall | MMD² |", "|---|---:|---:|---:|")
    for (i in seq_len(min(8, nrow(train_rows)))) {
      r <- train_rows[i, , drop = FALSE]
      lines <- c(lines, paste0("| ", r$run, " | ", format_number(r$best_train_combined), " | ", format_number(r$best_train_recall), " | ", format_number(r$best_train_mmd2), " |"))
    }
    lines <- c(lines, "")
  }

  writeLines(lines, out_path)
}

render_family_model_guidance <- function(guidance_df, family_df, out_path) {
  merged <- merge(guidance_df, family_df, by = c("logical_family_id", "dataset", "family"), all.x = TRUE, sort = FALSE)
  merged <- merged[order(merged$dataset, merged$family), , drop = FALSE]

  lines <- c(
    "# Family Model Guidance",
    "",
    "This report maps the validated and frontier model learnings from the Tencent and Alibaba training corpora back onto every logical trace family.",
    ""
  )

  for (i in seq_len(nrow(merged))) {
    row <- merged[i, , drop = FALSE]
    lines <- c(lines, paste0("## ", row$dataset, " / ", row$family), "")
    lines <- c(lines, paste0("- Closest learned anchor: ", row$closest_anchor, " (distance ", format_number(row$anchor_distance), ")"))
    lines <- c(lines, paste0("- Family kind: ", row$family_kind))
    lines <- c(lines, paste0("- Sampling: ", row$sampling_recommendation))
    lines <- c(lines, paste0("- Regime recipe: ", row$regime_recommendation))
    lines <- c(lines, paste0("- Char-file conditioning: ", if (isTRUE(row$char_file_conditioning)) "yes" else "no"))
    lines <- c(lines, paste0("- PCF status: ", row$pcf_status))
    lines <- c(lines, paste0("- Multi-scale critic status: ", row$multiscale_status))
    lines <- c(lines, paste0("- Mixed-type recovery status: ", row$mixed_type_status))
    lines <- c(lines, paste0("- Retrieval memory status: ", row$retrieval_status))
    if (nzchar(row$conditioning_candidates %||% "")) {
      lines <- c(lines, paste0("- Candidate conditioning additions: ", row$conditioning_candidates))
    }
    lines <- c(lines, "", "### Why", "", paste0("- ", row$rationale))
    if (nzchar(row$model_defaults %||% "")) {
      lines <- c(lines, "", "### Defaults", "")
      for (item in strsplit(row$model_defaults, ";", fixed = TRUE)[[1]]) {
        item <- trimws(item)
        if (nzchar(item)) {
          lines <- c(lines, paste0("- ", item))
        }
      }
    }
    if (nzchar(row$model_candidates %||% "")) {
      lines <- c(lines, "", "### Candidates", "")
      for (item in strsplit(row$model_candidates, ";", fixed = TRUE)[[1]]) {
        item <- trimws(item)
        if (nzchar(item)) {
          lines <- c(lines, paste0("- ", item))
        }
      }
    }
    lines <- c(lines, "")
  }

  writeLines(lines, out_path)
}

append_guidance_to_family_reports <- function(repo_sync_dir, guidance_df) {
  family_dir <- file.path(repo_sync_dir, "families")
  for (i in seq_len(nrow(guidance_df))) {
    row <- guidance_df[i, , drop = FALSE]
    path <- file.path(family_dir, paste0(row$logical_family_id, ".md"))
    if (!file.exists(path)) {
      next
    }
    existing <- readLines(path, warn = FALSE)
    marker <- which(existing == "## Model-Aware Guidance")
    if (length(marker) > 0) {
      existing <- existing[seq_len(marker[[1]] - 1L)]
    }
    section <- c(
      "",
      "## Model-Aware Guidance",
      "",
      paste0("- Closest learned anchor: ", row$closest_anchor, " (distance ", format_number(row$anchor_distance), ")"),
      paste0("- Sampling: ", row$sampling_recommendation),
      paste0("- Regime recipe: ", row$regime_recommendation),
      paste0("- Char-file conditioning: ", if (isTRUE(row$char_file_conditioning)) "yes" else "no"),
      paste0("- PCF: ", row$pcf_status),
      paste0("- Multi-scale critic: ", row$multiscale_status),
      paste0("- Mixed-type recovery: ", row$mixed_type_status),
      paste0("- Retrieval memory: ", row$retrieval_status),
      paste0("- Why: ", row$rationale)
    )
    if (nzchar(row$conditioning_candidates %||% "")) {
      section <- c(section, paste0("- Candidate conditioning additions: ", row$conditioning_candidates))
    }
    writeLines(c(existing, section), path)
  }
}

run_model_aware_analysis <- function(results_dir, repo_sync_dir, log_dir) {
  dir.create(repo_sync_dir, recursive = TRUE, showWarnings = FALSE)

  train_logs <- sort(list.files(log_dir, pattern = "^train_.*\\.log$", full.names = TRUE))
  eval_logs <- sort(list.files(log_dir, pattern = "^eval_.*\\.log$", full.names = TRUE))

  train_rows <- lapply(train_logs, parse_train_log)
  train_df <- if (length(train_rows) > 0) do.call(rbind, train_rows) else data.frame()

  eval_rows <- lapply(eval_logs, parse_eval_log)
  eval_df <- if (length(eval_rows) > 0) do.call(rbind, eval_rows) else data.frame()
  if (nrow(eval_df) > 0) {
    train_meta <- train_df
    train_meta$corpus_train <- train_meta$corpus
    train_meta$corpus <- NULL
    eval_df <- merge(eval_df, train_meta, by = "run", all.x = TRUE, sort = FALSE)
    eval_df$temporal_score <- rowSums(cbind(eval_df$dmdgen, eval_df$autocorr, eval_df$spectral, eval_df$hrc_mae), na.rm = TRUE)
  }

  family_df <- load_rollup_df(file.path(repo_sync_dir, "rollup.json"))
  corpus_summary <- summarize_corpora(train_df, eval_df)
  guidance_df <- build_family_guidance(family_df, corpus_summary)

  family_summary_csv <- file.path(repo_sync_dir, "families.csv")
  if (file.exists(family_summary_csv)) {
    family_summary <- read.csv(family_summary_csv, stringsAsFactors = FALSE)
    guidance_cols <- c(
      "closest_anchor",
      "anchor_distance",
      "sampling_recommendation",
      "regime_recommendation",
      "char_file_conditioning",
      "pcf_status",
      "multiscale_status",
      "mixed_type_status",
      "retrieval_status",
      "conditioning_candidates"
    )
    keep_cols <- setdiff(names(family_summary), c(guidance_cols, paste0(guidance_cols, ".x"), paste0(guidance_cols, ".y")))
    family_summary <- family_summary[, keep_cols, drop = FALSE]
    family_summary <- merge(
      family_summary,
      guidance_df[, c("logical_family_id", guidance_cols), drop = FALSE],
      by = "logical_family_id",
      all.x = TRUE,
      sort = FALSE
    )
    write.csv(family_summary, family_summary_csv, row.names = FALSE)
  }

  write.csv(train_df, file.path(repo_sync_dir, "model_train_runs.csv"), row.names = FALSE)
  write.csv(eval_df, file.path(repo_sync_dir, "model_eval_runs.csv"), row.names = FALSE)
  write.csv(corpus_summary, file.path(repo_sync_dir, "model_corpus_summary.csv"), row.names = FALSE)
  write.csv(guidance_df, file.path(repo_sync_dir, "family_model_guidance.csv"), row.names = FALSE)

  render_model_learnings(corpus_summary, eval_df, train_df, file.path(repo_sync_dir, "MODEL-LEARNINGS.md"))
  render_family_model_guidance(guidance_df, family_df, file.path(repo_sync_dir, "FAMILY-MODEL-GUIDANCE.md"))
  append_guidance_to_family_reports(repo_sync_dir, guidance_df)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) {
    stop("usage: run_model_aware_analysis.R <results_dir> <repo_sync_dir> <log_dir>")
  }
  run_model_aware_analysis(args[[1]], args[[2]], args[[3]])
}
