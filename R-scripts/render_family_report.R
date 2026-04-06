#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

format_number <- function(x, digits = 3) {
  if (is.null(x) || length(x) == 0) {
    return("N/A")
  }
  val <- suppressWarnings(as.numeric(x[[1]]))
  if (!is.finite(val)) {
    return("N/A")
  }
  format(round(val, digits), trim = TRUE, scientific = FALSE)
}

safe_read_csv <- function(path) {
  tryCatch(
    read.csv(path, stringsAsFactors = FALSE),
    error = function(e) data.frame()
  )
}

analysis_table_to_df <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(data.frame())
  }
  if (is.data.frame(x)) {
    return(x)
  }
  if (is.list(x) && !is.null(names(x))) {
    atomicish <- vapply(x, function(v) is.atomic(v) || is.null(v), logical(1))
    if (all(atomicish)) {
      return(data.frame(lapply(x, function(v) {
        lengths <- vapply(x, length, integer(1))
        max_len <- max(lengths, 1L)
        if (length(v) == 0) {
          return(rep(NA, max_len))
        }
        vals <- unlist(v, use.names = FALSE)
        if (length(vals) < max_len) {
          length(vals) <- max_len
        }
        vals
      }), stringsAsFactors = FALSE))
    }
  }
  if (is.list(x)) {
    rows <- lapply(x, function(item) {
      if (is.null(item) || length(item) == 0) {
        return(NULL)
      }
      if (is.data.frame(item)) {
        item
      } else if (is.list(item) && !is.null(names(item))) {
        as.data.frame(lapply(item, function(v) {
          if (length(v) == 0) {
            return(NA)
          }
          unlist(v, use.names = FALSE)[[1]]
        }), stringsAsFactors = FALSE)
      } else {
        tryCatch(as.data.frame(item, stringsAsFactors = FALSE), error = function(e) NULL)
      }
    })
    rows <- Filter(Negate(is.null), rows)
    if (length(rows) == 0) {
      return(data.frame())
    }
    return(do.call(rbind, rows))
  }
  data.frame()
}

render_family_report <- function(features_path, analysis_path, out_path) {
  df <- safe_read_csv(features_path)
  analysis <- jsonlite::fromJSON(analysis_path, simplifyVector = TRUE)
  metric_summary_path <- file.path(dirname(analysis_path), "metric_summary.csv")
  metric_summary <- if (file.exists(metric_summary_path) && file.info(metric_summary_path)$size > 0) safe_read_csv(metric_summary_path) else data.frame()

  lines <- c(
    paste("#", analysis$dataset, "/", analysis$family),
    "",
    paste("- Files:", analysis$files),
    paste("- Bytes:", format(analysis$bytes %||% 0, scientific = FALSE, trim = TRUE)),
    paste("- Formats:", paste(unlist(analysis$formats), collapse = ", ")),
    paste("- Parsers:", paste(unlist(analysis$parsers), collapse = ", ")),
    paste("- ML Use Cases:", paste(unlist(analysis$ml_use_cases %||% "unknown"), collapse = ", ")),
    paste("- Heterogeneity Score:", format_number(analysis$heterogeneity_score)),
    paste("- Suggested GAN Modes:", analysis$suggested_modes %||% 1),
    paste("- Split By Format:", if (isTRUE(analysis$split_by_format)) "yes" else "no"),
    ""
  )

  lines <- c(lines, "## Observations", "")
  for (note in unlist(analysis$observations %||% "No observations generated.")) {
    lines <- c(lines, paste0("- ", note))
  }

  lines <- c(lines, "", "## GAN Guidance", "")
  for (note in unlist(analysis$gan_guidance %||% "No GAN guidance generated.")) {
    lines <- c(lines, paste0("- ", note))
  }

  lines <- c(lines, "", "## Conditioning Audit", "", "| Item | Value |", "|---|---|")
  cond_drop <- analysis_table_to_df(analysis$conditioning_audit$near_constant)
  cond_add <- analysis_table_to_df(analysis$conditioning_audit$candidate_additions)
  cond_redundant <- analysis_table_to_df(analysis$conditioning_audit$redundant_pairs)
  recommended_additions <- character()
  if (nrow(cond_add) > 0 && "recommended" %in% names(cond_add)) {
    recommended_additions <- cond_add$metric[cond_add$recommended %in% TRUE]
  }
  lines <- c(lines, paste0("| Near-constant current conditioning features | ", if (nrow(cond_drop) > 0) paste(cond_drop$metric, collapse = ", ") else "none flagged", " |"))
  lines <- c(lines, paste0("| Recommended candidate additions | ", if (length(recommended_additions) > 0) paste(recommended_additions, collapse = ", ") else "none flagged", " |"))
  if (nrow(cond_redundant) > 0) {
    pair_text <- apply(head(cond_redundant, 4), 1, function(row) paste0(row[["metric_a"]], " vs ", row[["metric_b"]], " (", format_number(row[["correlation"]]), ")"))
    lines <- c(lines, paste0("| Highly redundant current pairs | ", paste(pair_text, collapse = "; "), " |"))
  } else {
    lines <- c(lines, "| Highly redundant current pairs | none flagged |")
  }

  lines <- c(lines, "", "## Format Breakdown", "", "| Format | Files | Parsers |", "|---|---:|---|")
  if (nrow(df) > 0 && "format" %in% names(df)) {
    for (fmt in sort(unique(df$format))) {
      subset <- df[df$format == fmt, , drop = FALSE]
      parser_values <- if ("parser" %in% names(subset)) paste(sort(unique(subset$parser)), collapse = ", ") else ""
      lines <- c(lines, paste0("| ", fmt, " | ", nrow(subset), " | ", parser_values, " |"))
    }
  }

  lines <- c(lines, "", "## Clustering And Regimes", "", "| Item | Value |", "|---|---|")
  if (!is.null(analysis$clusters$kmeans$selected_k)) {
    lines <- c(lines, paste0("| K-means selected K | ", analysis$clusters$kmeans$selected_k, " |"))
  }
  if (!is.null(analysis$clusters$kmeans_diagnostics$best_k)) {
    lines <- c(lines, paste0("| Best silhouette K | ", analysis$clusters$kmeans_diagnostics$best_k, " |"))
  }
  if (!is.null(analysis$clusters$mclust$components)) {
    lines <- c(lines, paste0("| Mclust components | ", analysis$clusters$mclust$components, " |"))
  }
  if (!is.null(analysis$clusters$dbscan$clusters)) {
    lines <- c(lines, paste0("| DBSCAN clusters | ", analysis$clusters$dbscan$clusters, " |"))
    lines <- c(lines, paste0("| DBSCAN noise fraction | ", format_number(analysis$clusters$dbscan$noise_fraction), " |"))
  }
  if (!is.null(analysis$regimes$changepoint_count)) {
    lines <- c(lines, paste0("| Ordered PC1 changepoints | ", analysis$regimes$changepoint_count, " |"))
  }
  if (!is.null(analysis$pca$pc1_variance)) {
    lines <- c(lines, paste0("| PCA variance explained by PC1 | ", format_number(analysis$pca$pc1_variance), " |"))
  }
  if (!is.null(analysis$regimes$tsfeatures$hurst)) {
    lines <- c(lines, paste0("| Hurst exponent on ordered PC1 | ", format_number(analysis$regimes$tsfeatures$hurst), " |"))
  }
  if (!is.null(analysis$temporal_sampling$block_random_distance_ratio)) {
    lines <- c(lines, paste0("| Block/random distance ratio | ", format_number(analysis$temporal_sampling$block_random_distance_ratio), " |"))
    lines <- c(lines, paste0("| Sampling recommendation | ", analysis$temporal_sampling$recommendation %||% "N/A", " |"))
  }

  kdiag <- analysis_table_to_df(analysis$clusters$kmeans_diagnostics$table)
  if (nrow(kdiag) > 0) {
    lines <- c(lines, "", "### K Selection", "", "| K | Within-SS | Silhouette |", "|---:|---:|---:|")
    for (i in seq_len(nrow(kdiag))) {
      row <- kdiag[i, , drop = FALSE]
      lines <- c(lines, paste0("| ", row$k, " | ", format_number(row$tot_withinss), " | ", format_number(row$silhouette), " |"))
    }
  }

  regime_df <- analysis_table_to_df(analysis$regime_attribution$transitions)
  if (nrow(regime_df) > 0) {
    lines <- c(lines, "", "## Regime Transition Drivers", "", "| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |", "|---|---|---:|---|---:|---|---:|")
    for (i in seq_len(nrow(regime_df))) {
      row <- regime_df[i, , drop = FALSE]
      lines <- c(
        lines,
        paste0(
          "| ", row$from_segment, " -> ", row$to_segment,
          " | ", row$top_driver_1 %||% "N/A",
          " | ", format_number(row$top_driver_1_effect),
          " | ", row$top_driver_2 %||% "N/A",
          " | ", format_number(row$top_driver_2_effect),
          " | ", row$top_driver_3 %||% "N/A",
          " | ", format_number(row$top_driver_3_effect),
          " |"
        )
      )
    }
  }

  lines <- c(lines, "", "## Strongest Correlations", "", "| Metric A | Metric B | Correlation |", "|---|---|---:|")
  corr_df <- analysis_table_to_df(analysis$top_correlations)
  if (nrow(corr_df) > 0) {
    for (i in seq_len(nrow(corr_df))) {
      row <- corr_df[i, , drop = FALSE]
      lines <- c(lines, paste0("| ", row$metric_a, " | ", row$metric_b, " | ", format_number(row$correlation), " |"))
    }
  } else {
    lines <- c(lines, "| N/A | N/A | N/A |")
  }

  lines <- c(lines, "", "## Metrics", "", "| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
  if (nrow(metric_summary) > 0) {
    display_metrics <- metric_summary[order(metric_summary$missing_frac, -abs(metric_summary$cv)), , drop = FALSE]
    display_metrics <- head(display_metrics, 18)
    for (i in seq_len(nrow(display_metrics))) {
      row <- display_metrics[i, , drop = FALSE]
      lines <- c(
        lines,
        paste0(
          "| ", row$metric,
          " | ", format_number(row$mean),
          " | ", format_number(row$median),
          " | ", format_number(row$cv),
          " | ", format_number(row$skewness),
          " | ", format_number(row$kurtosis),
          " | ", format_number(row$missing_frac),
          " | ", format_number(row$q10),
          " | ", format_number(row$q90),
          " |"
        )
      )
    }
  }

  lines <- c(lines, "", "## Outlier Files", "", "| rel_path | outlier_score | top drivers |", "|---|---:|---|")
  outlier_df <- analysis_table_to_df(analysis$outliers)
  outlier_decomp_df <- analysis_table_to_df(analysis$outlier_decomposition)
  if (nrow(outlier_df) > 0) {
    for (i in seq_len(min(8, nrow(outlier_df)))) {
      row <- outlier_df[i, , drop = FALSE]
      decomp <- outlier_decomp_df[outlier_decomp_df$rel_path == row$rel_path, , drop = FALSE]
      driver_text <- "N/A"
      if (nrow(decomp) > 0) {
        driver_text <- paste(
          paste0(decomp$top_feature_1[[1]] %||% "N/A", " (z=", format_number(decomp$top_feature_1_z[[1]]), ")"),
          paste0(decomp$top_feature_2[[1]] %||% "N/A", " (z=", format_number(decomp$top_feature_2_z[[1]]), ")"),
          sep = "; "
        )
      }
      lines <- c(lines, paste0("| ", row$rel_path, " | ", format_number(row$outlier_score), " | ", driver_text, " |"))
    }
  } else {
    lines <- c(lines, "| N/A | N/A | N/A |")
  }

  outlier_sensitivity_df <- analysis_table_to_df(analysis$outlier_sensitivity)
  if (nrow(outlier_sensitivity_df) > 0) {
    lines <- c(lines, "", "## Outlier Sensitivity", "", "| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |", "|---:|---|---:|---:|---:|")
    for (i in seq_len(min(10, nrow(outlier_sensitivity_df)))) {
      row <- outlier_sensitivity_df[i, , drop = FALSE]
      lines <- c(lines, paste0("| ", row$n_removed, " | ", row$metric, " | ", format_number(row$baseline_median), " | ", format_number(row$trimmed_median), " | ", format_number(row$relative_shift), " |"))
    }
  }

  lines <- c(lines, "", "## Notable Files", "", "| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |", "|---|---|---:|---:|---:|---:|")
  if (nrow(df) > 0 && all(c("rel_path", "format") %in% names(df))) {
    sortable <- df
    if ("burstiness_cv" %in% names(sortable)) {
      sortable$burstiness_cv[!is.finite(sortable$burstiness_cv)] <- -Inf
      sortable <- sortable[order(sortable$burstiness_cv, decreasing = TRUE), , drop = FALSE]
    }
    for (i in seq_len(min(8, nrow(sortable)))) {
      row <- sortable[i, , drop = FALSE]
      lines <- c(
        lines,
        paste0(
          "| ", row$rel_path,
          " | ", row$format,
          " | ", format_number(row$write_ratio),
          " | ", format_number(row$reuse_ratio),
          " | ", format_number(row$burstiness_cv),
          " | ", format_number(row$ts_duration),
          " |"
        )
      )
    }
  } else {
    lines <- c(lines, "| N/A | N/A | N/A | N/A | N/A | N/A |")
  }

  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, out_path)
}

if (sys.nframe() == 0) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required; run install_packages.R first")
  }
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) {
    stop("usage: render_family_report.R <features.csv> <analysis.json> <out.md>")
  }
  render_family_report(args[[1]], args[[2]], args[[3]])
}
