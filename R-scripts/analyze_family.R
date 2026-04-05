#!/usr/bin/env Rscript

numeric_columns <- c(
  "write_ratio",
  "reuse_ratio",
  "burstiness_cv",
  "iat_q50",
  "iat_q90",
  "obj_size_q50",
  "obj_size_q90",
  "tenant_unique",
  "idle_ratio",
  "total_iops_q50",
  "opcode_switch_ratio",
  "iat_lag1_autocorr",
  "forward_seek_ratio",
  "backward_seek_ratio"
)

safe_stat <- function(x, fn, default = NA_real_) {
  vals <- x[is.finite(x)]
  if (length(vals) == 0) {
    return(default)
  }
  fn(vals)
}

metric_summary <- function(df, metric) {
  vals <- suppressWarnings(as.numeric(df[[metric]]))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) {
    return(NULL)
  }
  list(
    metric = metric,
    n = length(vals),
    mean = mean(vals),
    median = stats::median(vals),
    min = min(vals),
    max = max(vals),
    sd = if (length(vals) > 1) stats::sd(vals) else 0
  )
}

family_observations <- function(df) {
  notes <- character()
  read_heavy <- safe_stat(1 - suppressWarnings(as.numeric(df$write_ratio)), mean)
  write_heavy <- safe_stat(suppressWarnings(as.numeric(df$write_ratio)), mean)
  locality <- safe_stat(suppressWarnings(as.numeric(df$reuse_ratio)), mean)
  burst <- safe_stat(suppressWarnings(as.numeric(df$burstiness_cv)), stats::median)
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
  if (length(notes) == 0) {
    notes <- "No single dominant behavioral note stood out from the sampled features."
  }
  notes
}

plot_family_metrics <- function(df, out_dir) {
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

  png(file.path(out_dir, "metric_pairs.png"), width = 1200, height = 900)
  par(mfrow = c(2, 2))
  safe_scatter(df$write_ratio, df$reuse_ratio, "write_ratio", "reuse_ratio", "Write vs Reuse")
  safe_scatter(df$burstiness_cv, df$iat_q50, "burstiness_cv", "iat_q50", "Burstiness vs IAT q50")
  safe_scatter(df$obj_size_q50, df$obj_size_q90, "obj_size_q50", "obj_size_q90", "Object Size Quantiles")
  hist_vals <- df$burstiness_cv[is.finite(df$burstiness_cv)]
  if (length(hist_vals) > 0) {
    hist(hist_vals, breaks = 20, main = "Burstiness Distribution", xlab = "burstiness_cv")
  } else {
    plot.new()
    title(main = "Burstiness Distribution")
    text(0.5, 0.5, "No finite values available")
  }
  dev.off()
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

  summaries <- Filter(Negate(is.null), lapply(numeric_columns, function(metric) metric_summary(df, metric)))
  summary_df <- if (length(summaries) > 0) {
    do.call(rbind, lapply(summaries, as.data.frame))
  } else {
    data.frame()
  }

  usable_numeric <- df[, intersect(numeric_columns, names(df)), drop = FALSE]
  usable_numeric[] <- lapply(usable_numeric, function(col) suppressWarnings(as.numeric(col)))
  keep <- colSums(is.finite(as.matrix(usable_numeric))) >= max(3, nrow(df) %/% 2)
  usable_numeric <- usable_numeric[, keep, drop = FALSE]

  pca_summary <- NULL
  cluster_summary <- NULL
  if (nrow(df) >= 4 && ncol(usable_numeric) >= 2) {
    complete_rows <- stats::complete.cases(usable_numeric)
    numeric_matrix <- usable_numeric[complete_rows, , drop = FALSE]
    if (ncol(numeric_matrix) > 0) {
      variances <- vapply(numeric_matrix, stats::var, numeric(1), na.rm = TRUE)
      numeric_matrix <- numeric_matrix[, is.finite(variances) & variances > 0, drop = FALSE]
    }
    if (nrow(numeric_matrix) >= 4) {
      pca <- stats::prcomp(numeric_matrix, center = TRUE, scale. = TRUE)
      pca_summary <- list(
        rotation = pca$rotation[, seq_len(min(3, ncol(pca$rotation))), drop = FALSE],
        importance = summary(pca)$importance[, seq_len(min(3, ncol(summary(pca)$importance))), drop = FALSE]
      )
      k <- min(3, nrow(numeric_matrix) - 1)
      if (k >= 2) {
        km <- stats::kmeans(numeric_matrix, centers = k, nstart = 10)
        cluster_summary <- list(
          centers = km$centers,
          sizes = km$size
        )
      }
    }
  }

  plot_family_metrics(df, out_dir)

  analysis <- list(
    dataset = df$dataset[[1]],
    family = df$family[[1]],
    logical_family_id = df$logical_family_id[[1]],
    files = nrow(df),
    bytes = sum(df$size_bytes, na.rm = TRUE),
    formats = sort(unique(df$format)),
    parsers = sort(unique(df$parser)),
    observations = family_observations(df),
    metrics = summaries,
    pca = pca_summary,
    clusters = cluster_summary
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

if (sys.nframe() == 0) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required; run install_packages.R first")
  }
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("usage: analyze_family.R <features.csv> <out_dir>")
  }
  analyze_family(args[[1]], args[[2]])
}
