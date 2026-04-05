#!/usr/bin/env Rscript

format_number <- function(x, digits = 3) {
  if (is.null(x) || length(x) == 0 || !is.finite(x)) {
    return("N/A")
  }
  format(round(x, digits), trim = TRUE, scientific = FALSE)
}

render_family_report <- function(features_path, analysis_path, out_path) {
  safe_read_csv <- function(path) {
    tryCatch(
      read.csv(path, stringsAsFactors = FALSE),
      error = function(e) data.frame()
    )
  }

  df <- safe_read_csv(features_path)
  analysis <- jsonlite::fromJSON(analysis_path, simplifyVector = FALSE)
  metric_summary_path <- file.path(dirname(analysis_path), "metric_summary.csv")
  metric_summary <- if (file.exists(metric_summary_path) && file.info(metric_summary_path)$size > 0) {
    safe_read_csv(metric_summary_path)
  } else {
    data.frame()
  }

  lines <- c(
    paste("#", analysis$dataset, "/", analysis$family),
    "",
    paste("- Files:", analysis$files),
    paste("- Bytes:", format(if ("size_bytes" %in% names(df)) sum(df$size_bytes, na.rm = TRUE) else analysis$bytes %||% 0, scientific = FALSE, trim = TRUE)),
    paste("- Formats:", paste(sort(unique(if ("format" %in% names(df)) df$format else unlist(analysis$formats))), collapse = ", ")),
    paste("- Parsers:", paste(sort(unique(if ("parser" %in% names(df)) df$parser else unlist(analysis$parsers))), collapse = ", ")),
    ""
  )

  lines <- c(lines, "## Observations", "")
  for (note in unlist(analysis$observations)) {
    lines <- c(lines, paste("- ", note, sep = ""))
  }
  lines <- c(lines, "", "## Format Breakdown", "", "| Format | Files | Parsers |", "|---|---:|---|")
  for (fmt in sort(unique(if ("format" %in% names(df)) df$format else unlist(analysis$formats)))) {
    subset <- df[df$format == fmt, , drop = FALSE]
    parser_values <- if ("parser" %in% names(subset)) paste(sort(unique(subset$parser)), collapse = ", ") else paste(sort(unique(unlist(analysis$parsers))), collapse = ", ")
    lines <- c(lines, paste0("| ", fmt, " | ", nrow(subset), " | ", parser_values, " |"))
  }

  lines <- c(lines, "", "## Metrics", "", "| Metric | Mean | Median | Min | Max | SD |", "|---|---:|---:|---:|---:|---:|")
  if (nrow(metric_summary) > 0) {
    for (i in seq_len(nrow(metric_summary))) {
      row <- metric_summary[i, ]
      lines <- c(
        lines,
        paste0(
          "| ", row$metric,
          " | ", format_number(row$mean),
          " | ", format_number(row$median),
          " | ", format_number(row$min),
          " | ", format_number(row$max),
          " | ", format_number(row$sd),
          " |"
        )
      )
    }
  }

  lines <- c(lines, "", "## Notable Files", "", "| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |", "|---|---|---:|---:|---:|")
  if (nrow(df) > 0 && all(c("rel_path", "format", "write_ratio", "reuse_ratio", "burstiness_cv") %in% names(df))) {
    sortable <- df
    sortable$burstiness_cv[!is.finite(sortable$burstiness_cv)] <- -Inf
    sortable <- sortable[order(sortable$burstiness_cv, decreasing = TRUE), ]
    for (i in seq_len(min(5, nrow(sortable)))) {
      row <- sortable[i, ]
      lines <- c(
        lines,
        paste0(
          "| ", row$rel_path,
          " | ", row$format,
          " | ", format_number(row$write_ratio),
          " | ", format_number(row$reuse_ratio),
          " | ", format_number(row$burstiness_cv),
          " |"
        )
      )
    }
  } else {
    lines <- c(lines, "| No per-file rows available | | | | |")
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
