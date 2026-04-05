#!/usr/bin/env Rscript

required_packages <- c("jsonlite")
optional_packages <- c("changepoint", "dbscan", "e1071", "mclust", "moments", "forecast", "tsfeatures", "corrplot")

package_available <- function(pkg) {
  requireNamespace(pkg, quietly = TRUE)
}

for (pkg in required_packages) {
  if (!package_available(pkg)) {
    stop(pkg, " is required; run install_packages.R first")
  }
}

numeric_columns <- c(
  "sample_records",
  "size_bytes",
  "ts_duration",
  "sample_record_rate",
  "iat_zero_ratio",
  "iat_lag1_autocorr",
  "burstiness_cv",
  "iat_min",
  "iat_mean",
  "iat_std",
  "iat_q50",
  "iat_q90",
  "iat_q99",
  "obj_size_min",
  "obj_size_mean",
  "obj_size_std",
  "obj_size_q50",
  "obj_size_q90",
  "obj_size_q99",
  "write_ratio",
  "opcode_switch_ratio",
  "reuse_ratio",
  "forward_seek_ratio",
  "backward_seek_ratio",
  "signed_stride_lag1_autocorr",
  "abs_stride_mean",
  "abs_stride_std",
  "abs_stride_q50",
  "abs_stride_q90",
  "abs_stride_q99",
  "tenant_unique",
  "tenant_top1_share",
  "tenant_top10_share",
  "object_unique",
  "object_top1_share",
  "object_top10_share",
  "response_time_mean",
  "response_time_std",
  "response_time_q50",
  "response_time_q90",
  "response_time_q99",
  "lcs_version",
  "ttl_present",
  "feature_field_count",
  "sampling_interval_seconds",
  "sampling_interval_q50",
  "sampling_interval_q90",
  "sampling_interval_q99",
  "read_iops_mean",
  "read_iops_q50",
  "read_iops_q90",
  "read_iops_q99",
  "write_iops_mean",
  "write_iops_q50",
  "write_iops_q90",
  "write_iops_q99",
  "total_iops_mean",
  "total_iops_q50",
  "total_iops_q90",
  "total_iops_q99",
  "read_bw_mean",
  "read_bw_q50",
  "read_bw_q90",
  "write_bw_mean",
  "write_bw_q50",
  "write_bw_q90",
  "total_bw_mean",
  "total_bw_q50",
  "total_bw_q90",
  "disk_usage_mean",
  "disk_usage_q50",
  "disk_usage_q90",
  "idle_ratio",
  "write_share_iops_mean",
  "total_iops_lag1_autocorr",
  "disk_usage_lag1_autocorr",
  "schema_column_count",
  "schema_numeric_cols",
  "schema_mixed_cols",
  "schema_high_cardinality_cols",
  "first_numeric_monotone_ratio",
  "first_numeric_diff_mean",
  "first_numeric_diff_std",
  "first_numeric_diff_q50",
  "first_numeric_diff_q90"
)

format_number <- function(x, digits = 3) {
  if (is.null(x) || length(x) == 0 || all(!is.finite(x))) {
    return("N/A")
  }
  format(round(x[[1]], digits), trim = TRUE, scientific = FALSE)
}

safe_stat <- function(x, fn, default = NA_real_) {
  vals <- suppressWarnings(as.numeric(x))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) {
    return(default)
  }
  fn(vals)
}

coerce_numeric_frame <- function(df) {
  cols <- intersect(numeric_columns, names(df))
  out <- df[, cols, drop = FALSE]
  out[] <- lapply(out, function(col) suppressWarnings(as.numeric(col)))
  out
}

metric_summary_row <- function(df, metric) {
  vals <- suppressWarnings(as.numeric(df[[metric]]))
  finite <- vals[is.finite(vals)]
  if (length(finite) == 0) {
    return(NULL)
  }
  mad_val <- if (length(finite) > 1) stats::mad(finite, center = stats::median(finite), constant = 1) else 0
  sd_val <- if (length(finite) > 1) stats::sd(finite) else 0
  mean_val <- mean(finite)
  skew <- if (package_available("e1071") && length(finite) > 2) e1071::skewness(finite, na.rm = TRUE, type = 2) else if (package_available("moments") && length(finite) > 2) moments::skewness(finite, na.rm = TRUE) else NA_real_
  kurt <- if (package_available("e1071") && length(finite) > 3) e1071::kurtosis(finite, na.rm = TRUE, type = 2) else if (package_available("moments") && length(finite) > 3) moments::kurtosis(finite, na.rm = TRUE) else NA_real_
  data.frame(
    metric = metric,
    n = length(finite),
    missing_frac = mean(!is.finite(vals)),
    mean = mean_val,
    median = stats::median(finite),
    mad = mad_val,
    sd = sd_val,
    cv = if (isTRUE(abs(mean_val) > 1e-12)) sd_val / abs(mean_val) else NA_real_,
    skewness = skew,
    kurtosis = kurt,
    q10 = stats::quantile(finite, 0.10, names = FALSE, type = 7),
    q25 = stats::quantile(finite, 0.25, names = FALSE, type = 7),
    q75 = stats::quantile(finite, 0.75, names = FALSE, type = 7),
    q90 = stats::quantile(finite, 0.90, names = FALSE, type = 7),
    min = min(finite),
    max = max(finite),
    stringsAsFactors = FALSE
  )
}

metric_summaries <- function(df) {
  Filter(Negate(is.null), lapply(intersect(numeric_columns, names(df)), function(metric) metric_summary_row(df, metric)))
}

top_correlation_pairs <- function(cor_mat, limit = 8) {
  if (is.null(cor_mat) || ncol(cor_mat) < 2) {
    return(data.frame())
  }
  pairs <- list()
  idx <- 1L
  for (i in seq_len(ncol(cor_mat) - 1)) {
    for (j in seq.int(i + 1L, ncol(cor_mat))) {
      val <- cor_mat[i, j]
      if (!is.finite(val)) {
        next
      }
      pairs[[idx]] <- data.frame(
        metric_a = colnames(cor_mat)[[i]],
        metric_b = colnames(cor_mat)[[j]],
        correlation = val,
        abs_correlation = abs(val),
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  if (length(pairs) == 0) {
    return(data.frame())
  }
  pair_df <- do.call(rbind, pairs)
  pair_df <- pair_df[order(pair_df$abs_correlation, decreasing = TRUE), , drop = FALSE]
  head(pair_df, limit)
}

compute_heterogeneity <- function(metric_summary_df) {
  if (nrow(metric_summary_df) == 0) {
    return(NA_real_)
  }
  vals <- metric_summary_df$cv
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) {
    return(NA_real_)
  }
  stats::median(pmin(vals, 25))
}

make_pca_bundle <- function(numeric_matrix, sample_ids, out_dir) {
  if (nrow(numeric_matrix) < 4 || ncol(numeric_matrix) < 2) {
    return(NULL)
  }
  pca <- stats::prcomp(numeric_matrix, center = TRUE, scale. = TRUE)
  scores <- as.data.frame(pca$x[, seq_len(min(3, ncol(pca$x))), drop = FALSE])
  scores$rel_path <- sample_ids
  importance <- summary(pca)$importance

  png(file.path(out_dir, "pca_scree.png"), width = 1000, height = 700)
  scree <- importance[2, ]
  barplot(scree, main = "PCA Variance Explained", ylab = "Proportion", xlab = "PC")
  dev.off()

  if (ncol(scores) >= 2) {
    png(file.path(out_dir, "pca_scatter.png"), width = 1000, height = 800)
    plot(scores[[1]], scores[[2]], xlab = names(scores)[1], ylab = names(scores)[2], main = "PCA Scatter")
    dev.off()
  }

  list(
    model = pca,
    scores = scores,
    importance = importance
  )
}

compute_clusters <- function(numeric_matrix, sample_ids) {
  if (nrow(numeric_matrix) < 8 || ncol(numeric_matrix) < 2) {
    return(list(summary = NULL, assignments = data.frame()))
  }

  sample_n <- min(nrow(numeric_matrix), 4000L)
  if (nrow(numeric_matrix) > sample_n) {
    idx <- sort(sample(seq_len(nrow(numeric_matrix)), sample_n))
    cluster_matrix <- numeric_matrix[idx, , drop = FALSE]
    cluster_ids <- sample_ids[idx]
  } else {
    cluster_matrix <- numeric_matrix
    cluster_ids <- sample_ids
  }

  out <- list(summary = list(), assignments = data.frame(rel_path = cluster_ids, stringsAsFactors = FALSE))

  k <- min(6L, nrow(cluster_matrix) - 1L)
  if (k >= 2L) {
    km <- stats::kmeans(cluster_matrix, centers = k, nstart = 20)
    out$summary$kmeans <- list(centers = km$centers, sizes = km$size, tot_withinss = km$tot.withinss)
    out$assignments$kmeans_cluster <- km$cluster
  }

  if (package_available("mclust")) {
    mc <- tryCatch(mclust::Mclust(cluster_matrix, G = 1:min(6L, nrow(cluster_matrix) - 1L), verbose = FALSE), error = function(e) NULL)
    if (!is.null(mc)) {
      out$summary$mclust <- list(
        components = mc$G,
        model_name = mc$modelName,
        bic = max(mc$bic, na.rm = TRUE),
        uncertainty_mean = mean(mc$uncertainty, na.rm = TRUE)
      )
      out$assignments$mclust_cluster <- mc$classification
      out$assignments$mclust_uncertainty <- mc$uncertainty
    }
  }

  if (package_available("dbscan")) {
    min_pts <- max(4L, min(20L, floor(log(nrow(cluster_matrix)) * 3)))
    eps <- tryCatch(dbscan::kNNdist(cluster_matrix, k = min_pts), error = function(e) numeric())
    eps_val <- if (length(eps) > 0) stats::quantile(eps, 0.90, names = FALSE, na.rm = TRUE) else NA_real_
    if (is.finite(eps_val) && eps_val > 0) {
      db <- tryCatch(dbscan::dbscan(cluster_matrix, eps = eps_val, minPts = min_pts), error = function(e) NULL)
      if (!is.null(db)) {
        out$summary$dbscan <- list(
          eps = eps_val,
          min_pts = min_pts,
          clusters = max(db$cluster),
          noise_fraction = mean(db$cluster == 0)
        )
        out$assignments$dbscan_cluster <- db$cluster
      }
    }
  }

  out
}

compute_outliers <- function(pca_bundle, limit = 8) {
  if (is.null(pca_bundle)) {
    return(data.frame())
  }
  scores <- pca_bundle$scores
  pc_cols <- setdiff(names(scores), "rel_path")
  if (length(pc_cols) == 0) {
    return(data.frame())
  }
  score_matrix <- as.matrix(scores[, pc_cols, drop = FALSE])
  vars <- apply(score_matrix, 2, stats::var)
  keep <- is.finite(vars) & vars > 0
  score_matrix <- score_matrix[, keep, drop = FALSE]
  if (ncol(score_matrix) == 0) {
    return(data.frame())
  }
  center <- colMeans(score_matrix)
  cov_mat <- stats::cov(score_matrix)
  distances <- tryCatch(stats::mahalanobis(score_matrix, center = center, cov = cov_mat), error = function(e) rowSums(scale(score_matrix)^2, na.rm = TRUE))
  out <- data.frame(rel_path = scores$rel_path, outlier_score = distances, stringsAsFactors = FALSE)
  out <- out[order(out$outlier_score, decreasing = TRUE), , drop = FALSE]
  head(out, limit)
}

compute_regimes <- function(df, pca_bundle) {
  if (is.null(pca_bundle)) {
    return(NULL)
  }
  if (!("ts_start" %in% names(df))) {
    return(NULL)
  }
  ordered <- df[is.finite(df$ts_start), c("rel_path", "ts_start"), drop = FALSE]
  if (nrow(ordered) < 8) {
    return(NULL)
  }
  ordered <- ordered[order(ordered$ts_start, ordered$rel_path), , drop = FALSE]
  score_df <- pca_bundle$scores
  merged <- merge(ordered, score_df, by = "rel_path", all.x = FALSE, all.y = FALSE)
  if (nrow(merged) < 8 || !("PC1" %in% names(merged))) {
    return(NULL)
  }
  series <- merged$PC1
  regime <- list(
    ordered_files = nrow(merged),
    series_mean = mean(series),
    series_sd = stats::sd(series)
  )

  if (package_available("changepoint")) {
    cp <- tryCatch(changepoint::cpt.meanvar(series, method = "PELT", penalty = "SIC"), error = function(e) NULL)
    if (!is.null(cp)) {
      cps <- changepoint::cpts(cp)
      regime$changepoint_count <- length(cps)
      regime$changepoints <- as.integer(cps)
    }
  }

  if (package_available("tsfeatures")) {
    regime$tsfeatures <- tryCatch(
      {
        feats <- tsfeatures::tsfeatures(stats::ts(series), c("entropy", "lumpiness", "stability", "hurst", "flat_spots"))
        as.list(feats[1, , drop = TRUE])
      },
      error = function(e) NULL
    )
  }

  regime
}

build_gan_guidance <- function(df, heterogeneity_score, cluster_summary, regime_summary, top_corr, outliers) {
  notes <- character()
  read_heavy <- safe_stat(1 - df$write_ratio, mean)
  write_heavy <- safe_stat(df$write_ratio, mean)
  locality <- safe_stat(df$reuse_ratio, stats::median)
  burst <- safe_stat(df$burstiness_cv, stats::median)
  tenant_mix <- safe_stat(df$tenant_unique, stats::median)
  formats <- sort(unique(df$format))
  parsers <- sort(unique(df$parser))

  if (length(formats) > 1 || length(parsers) > 1) {
    notes <- c(notes, "Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.")
  }
  if (is.finite(heterogeneity_score) && heterogeneity_score > 2.5) {
    notes <- c(notes, "High cross-file heterogeneity; favor regime conditioning or multiple family-specific GAN runs over a single unconditional model.")
  }
  if (!is.null(cluster_summary$mclust) && is.finite(cluster_summary$mclust$components) && cluster_summary$mclust$components >= 3) {
    notes <- c(notes, paste0("Gaussian mixture analysis suggests about ", cluster_summary$mclust$components, " workload modes inside this family."))
  }
  if (!is.null(regime_summary$changepoint_count) && regime_summary$changepoint_count >= 1) {
    notes <- c(notes, paste0("Ordered PC1 changepoints suggest ", regime_summary$changepoint_count + 1, " regimes when files are ordered by trace start time."))
  }
  if (is.finite(read_heavy) && read_heavy > 0.9) {
    notes <- c(notes, "Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.")
  }
  if (is.finite(write_heavy) && write_heavy > 0.6) {
    notes <- c(notes, "Write pressure is material; preserve write bursts and opcode transitions in conditioning.")
  }
  if (is.finite(locality) && locality > 0.5) {
    notes <- c(notes, "Reuse/locality is a major axis here; locality-aware losses and conditioning should matter.")
  }
  if (is.finite(burst) && burst > 10) {
    notes <- c(notes, "Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.")
  }
  if (is.finite(tenant_mix) && tenant_mix > 8) {
    notes <- c(notes, "Tenant diversity is high; tenant/context conditioning is likely useful.")
  }
  if (nrow(top_corr) > 0) {
    first_pair <- top_corr[1, , drop = FALSE]
    notes <- c(notes, paste0("Strongest feature coupling in this pass: ", first_pair$metric_a[[1]], " vs ", first_pair$metric_b[[1]], " (corr=", format_number(first_pair$correlation[[1]], 2), ")."))
  }
  if (nrow(outliers) > 0) {
    notes <- c(notes, "A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.")
  }
  if (length(notes) == 0) {
    notes <- "Family looks comparatively homogeneous in the extracted feature space."
  }
  notes
}

family_observations <- function(df, metric_summary_df, cluster_summary, regime_summary) {
  notes <- character()
  read_heavy <- safe_stat(1 - df$write_ratio, mean)
  write_heavy <- safe_stat(df$write_ratio, mean)
  locality <- safe_stat(df$reuse_ratio, mean)
  burst <- safe_stat(df$burstiness_cv, stats::median)
  if (is.finite(read_heavy) && read_heavy > 0.8) {
    notes <- c(notes, "Predominantly read-heavy.")
  }
  if (is.finite(write_heavy) && write_heavy > 0.6) {
    notes <- c(notes, "Substantial write pressure across sampled files.")
  }
  if (is.finite(locality) && locality > 0.5) {
    notes <- c(notes, "High temporal locality / reuse.")
  }
  if (is.finite(locality) && locality < 0.05) {
    notes <- c(notes, "Very weak short-window reuse.")
  }
  if (is.finite(burst) && burst > 10) {
    notes <- c(notes, "Highly bursty arrivals.")
  }
  if (!is.null(cluster_summary$mclust) && cluster_summary$mclust$components >= 3) {
    notes <- c(notes, "Multiple workload modes are visible in the feature space.")
  }
  if (!is.null(regime_summary$changepoint_count) && regime_summary$changepoint_count >= 1) {
    notes <- c(notes, "Ordered feature trajectories show regime boundaries.")
  }
  if (nrow(metric_summary_df) > 0 && sum(metric_summary_df$missing_frac < 0.5, na.rm = TRUE) < 5) {
    notes <- c(notes, "Much of the family is only partially represented in numeric features; interpret structured-table metrics separately.")
  }
  if (length(notes) == 0) {
    notes <- "No single dominant behavioral note stood out from the sampled features."
  }
  notes
}

plot_family_metrics <- function(df, out_dir, cor_mat = NULL, pca_bundle = NULL) {
  safe_scatter <- function(x, y, xlab, ylab, main) {
    finite <- is.finite(x) & is.finite(y)
    if (!any(finite)) {
      plot.new()
      title(main = main)
      text(0.5, 0.5, "No finite values available")
    } else {
      plot(x[finite], y[finite], xlab = xlab, ylab = ylab, main = main)
    }
  }

  png(file.path(out_dir, "metric_pairs.png"), width = 1400, height = 1000)
  par(mfrow = c(2, 2))
  safe_scatter(df$write_ratio, df$reuse_ratio, "write_ratio", "reuse_ratio", "Write vs Reuse")
  safe_scatter(df$burstiness_cv, df$iat_q50, "burstiness_cv", "iat_q50", "Burstiness vs IAT q50")
  safe_scatter(df$obj_size_q50, df$obj_size_q90, "obj_size_q50", "obj_size_q90", "Object Size Quantiles")
  hist_vals <- suppressWarnings(as.numeric(df$burstiness_cv))
  hist_vals <- hist_vals[is.finite(hist_vals)]
  if (length(hist_vals) > 0) {
    hist(hist_vals, breaks = 20, main = "Burstiness Distribution", xlab = "burstiness_cv")
  } else {
    plot.new()
    title(main = "Burstiness Distribution")
    text(0.5, 0.5, "No finite values available")
  }
  dev.off()

  if (!is.null(cor_mat) && ncol(cor_mat) >= 2) {
    png(file.path(out_dir, "correlation_heatmap.png"), width = 1200, height = 1200)
    if (package_available("corrplot")) {
      corrplot::corrplot(cor_mat, method = "color", tl.cex = 0.7, type = "upper")
    } else {
      heatmap(cor_mat, symm = TRUE)
    }
    dev.off()
  }

  if (!is.null(pca_bundle) && "PC1" %in% names(pca_bundle$scores) && "PC2" %in% names(pca_bundle$scores)) {
    png(file.path(out_dir, "pc_order.png"), width = 1200, height = 700)
    plot(seq_len(nrow(pca_bundle$scores)), pca_bundle$scores$PC1, type = "l", xlab = "Sample index", ylab = "PC1", main = "PC1 Across Samples")
    dev.off()
  }
}

sanitize_json <- function(x) {
  if (is.list(x)) {
    return(lapply(x, sanitize_json))
  }
  if (is.data.frame(x)) {
    out <- x
    out[] <- lapply(out, sanitize_json)
    return(out)
  }
  if (is.matrix(x)) {
    out <- x
    out[!is.finite(out)] <- NA_real_
    return(out)
  }
  if (is.numeric(x)) {
    out <- x
    out[!is.finite(out)] <- NA_real_
    return(out)
  }
  x
}

analyze_family <- function(features_path, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  df <- read.csv(features_path, stringsAsFactors = FALSE)

  summaries <- metric_summaries(df)
  summary_df <- if (length(summaries) > 0) do.call(rbind, summaries) else data.frame()

  usable_numeric <- coerce_numeric_frame(df)
  keep <- colSums(is.finite(as.matrix(usable_numeric))) >= max(4L, floor(nrow(df) * 0.5))
  usable_numeric <- usable_numeric[, keep, drop = FALSE]

  numeric_matrix <- usable_numeric[stats::complete.cases(usable_numeric), , drop = FALSE]
  numeric_ids <- df$rel_path[stats::complete.cases(usable_numeric)]
  if (ncol(numeric_matrix) > 0) {
    variances <- vapply(numeric_matrix, stats::var, numeric(1), na.rm = TRUE)
    numeric_matrix <- numeric_matrix[, is.finite(variances) & variances > 0, drop = FALSE]
  }

  cor_mat <- NULL
  top_corr <- data.frame()
  if (ncol(usable_numeric) >= 2) {
    cor_mat <- suppressWarnings(stats::cor(usable_numeric, use = "pairwise.complete.obs"))
    top_corr <- top_correlation_pairs(cor_mat)
  }

  pca_bundle <- if (nrow(numeric_matrix) >= 4 && ncol(numeric_matrix) >= 2) make_pca_bundle(numeric_matrix, numeric_ids, out_dir) else NULL
  cluster_results <- if (nrow(numeric_matrix) >= 8 && ncol(numeric_matrix) >= 2) compute_clusters(numeric_matrix, numeric_ids) else list(summary = NULL, assignments = data.frame())
  regime_summary <- compute_regimes(df, pca_bundle)
  outliers <- compute_outliers(pca_bundle)
  heterogeneity_score <- compute_heterogeneity(summary_df)

  plot_family_metrics(df, out_dir, cor_mat = cor_mat, pca_bundle = pca_bundle)

  if (nrow(cluster_results$assignments) > 0) {
    write.csv(cluster_results$assignments, file.path(out_dir, "cluster_assignments.csv"), row.names = FALSE)
  }
  if (nrow(top_corr) > 0) {
    write.csv(top_corr, file.path(out_dir, "top_correlations.csv"), row.names = FALSE)
  }
  if (nrow(outliers) > 0) {
    write.csv(outliers, file.path(out_dir, "outliers.csv"), row.names = FALSE)
  }

  observations <- family_observations(df, summary_df, cluster_results$summary %||% list(), regime_summary %||% list())
  gan_guidance <- build_gan_guidance(
    df,
    heterogeneity_score = heterogeneity_score,
    cluster_summary = cluster_results$summary %||% list(),
    regime_summary = regime_summary %||% list(),
    top_corr = top_corr,
    outliers = outliers
  )

  split_by_format <- length(unique(df$format)) > 1 || length(unique(df$parser)) > 1
  suggested_modes <- max(
    1L,
    min(8L, if (!is.null(cluster_results$summary$mclust$components)) as.integer(cluster_results$summary$mclust$components) else 1L),
    min(8L, if (!is.null(regime_summary$changepoint_count)) as.integer(regime_summary$changepoint_count) + 1L else 1L)
  )

  analysis <- list(
    dataset = df$dataset[[1]],
    family = df$family[[1]],
    logical_family_id = df$logical_family_id[[1]],
    files = nrow(df),
    bytes = sum(df$size_bytes, na.rm = TRUE),
    formats = sort(unique(df$format)),
    parsers = sort(unique(df$parser)),
    ml_use_cases = sort(unique(df$ml_use_case)),
    observations = observations,
    gan_guidance = gan_guidance,
    heterogeneity_score = heterogeneity_score,
    suggested_modes = suggested_modes,
    split_by_format = split_by_format,
    metrics = summaries,
    top_correlations = top_corr,
    outliers = outliers,
    pca = if (!is.null(pca_bundle)) list(
      pc1_variance = pca_bundle$importance[2, 1],
      importance = pca_bundle$importance[, seq_len(min(4, ncol(pca_bundle$importance))), drop = FALSE],
      rotation = pca_bundle$model$rotation[, seq_len(min(4, ncol(pca_bundle$model$rotation))), drop = FALSE]
    ) else NULL,
    clusters = cluster_results$summary,
    regimes = regime_summary
  )

  analysis <- sanitize_json(analysis)
  jsonlite::write_json(
    analysis,
    file.path(out_dir, "analysis.json"),
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null",
    na = "null"
  )
  write.csv(summary_df, file.path(out_dir, "metric_summary.csv"), row.names = FALSE)
  invisible(analysis)
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("usage: analyze_family.R <features.csv> <out_dir>")
  }
  analyze_family(args[[1]], args[[2]])
}
