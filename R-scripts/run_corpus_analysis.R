#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required; run install_packages.R first")
}

script_arg <- commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))]
script_path <- sub("^--file=", "", script_arg[[1]] %||% "")
script_dir <- if (nzchar(script_path)) dirname(normalizePath(script_path)) else getwd()

source(file.path(script_dir, "extract_family_features.R"))
source(file.path(script_dir, "analyze_family.R"))
source(file.path(script_dir, "render_family_report.R"))

detect_worker_cap <- function() {
  cmd <- suppressWarnings(system("pgrep -af train.py", intern = TRUE, ignore.stderr = TRUE))
  if (length(cmd) > 0) {
    return(6L)
  }
  12L
}

format_number <- function(x, digits = 3) {
  val <- suppressWarnings(as.numeric(x[[1]] %||% NA_real_))
  if (!is.finite(val)) {
    return("N/A")
  }
  format(round(val, digits), trim = TRUE, scientific = FALSE)
}

read_analysis_json <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

flatten_analysis_summary <- function(analysis) {
  data.frame(
    logical_family_id = analysis$logical_family_id,
    dataset = analysis$dataset,
    family = analysis$family,
    files = analysis$files,
    bytes = analysis$bytes,
    formats = paste(analysis$formats, collapse = ","),
    parsers = paste(analysis$parsers, collapse = ","),
    ml_use_cases = paste(analysis$ml_use_cases %||% character(), collapse = ","),
    heterogeneity_score = analysis$heterogeneity_score %||% NA_real_,
    suggested_modes = analysis$suggested_modes %||% 1L,
    split_by_format = isTRUE(analysis$split_by_format),
    mclust_components = analysis$clusters$mclust$components %||% NA_real_,
    dbscan_clusters = analysis$clusters$dbscan$clusters %||% NA_real_,
    changepoint_count = analysis$regimes$changepoint_count %||% NA_real_,
    top_gan_guidance = paste(head(analysis$gan_guidance %||% character(), 3), collapse = " "),
    stringsAsFactors = FALSE
  )
}

build_rollup <- function(features, family_summary) {
  groups <- list()
  for (i in seq_len(nrow(family_summary))) {
    row <- family_summary[i, , drop = FALSE]
    family_id <- row$logical_family_id[[1]]
    subset <- features[features$logical_family_id == family_id, , drop = FALSE]
    groups[[family_id]] <- list(
      dataset = row$dataset[[1]],
      family = row$family[[1]],
      files = row$files[[1]],
      bytes = row$bytes[[1]],
      formats = strsplit(row$formats[[1]], ",", fixed = TRUE)[[1]],
      parsers = strsplit(row$parsers[[1]], ",", fixed = TRUE)[[1]],
      ml_use_cases = strsplit(row$ml_use_cases[[1]], ",", fixed = TRUE)[[1]],
      heterogeneity_score = row$heterogeneity_score[[1]],
      suggested_modes = row$suggested_modes[[1]],
      split_by_format = isTRUE(row$split_by_format[[1]]),
      mclust_components = row$mclust_components[[1]],
      dbscan_clusters = row$dbscan_clusters[[1]],
      changepoint_count = row$changepoint_count[[1]],
      gan_guidance = row$top_gan_guidance[[1]],
      write_ratio = stats::median(subset$write_ratio, na.rm = TRUE),
      reuse_ratio = stats::median(subset$reuse_ratio, na.rm = TRUE),
      burstiness_cv = stats::median(subset$burstiness_cv, na.rm = TRUE),
      iat_q50 = stats::median(subset$iat_q50, na.rm = TRUE),
      obj_size_q50 = stats::median(subset$obj_size_q50, na.rm = TRUE),
      tenant_unique = stats::median(subset$tenant_unique, na.rm = TRUE)
    )
  }
  groups
}

render_readme <- function(family_summary, out_path) {
  lines <- c(
    "# Trace Family Characterizations",
    "",
    paste0("**Total logical families:** ", nrow(family_summary), "  "),
    paste0("**Families suggesting multiple GAN modes:** ", sum(family_summary$suggested_modes > 1, na.rm = TRUE), "  "),
    "",
    "| Family | Dataset | Files | Size | Heterogeneity | Modes | Split Format | Guidance |",
    "|---|---|---:|---:|---:|---:|---|---|"
  )
  ordered <- family_summary[order(family_summary$heterogeneity_score, decreasing = TRUE, na.last = TRUE), , drop = FALSE]
  for (i in seq_len(nrow(ordered))) {
    row <- ordered[i, , drop = FALSE]
    size_gb <- row$bytes[[1]] / (1024 ^ 3)
    lines <- c(
      lines,
      paste0(
        "| ", row$family[[1]],
        " | ", row$dataset[[1]],
        " | ", row$files[[1]],
        " | ", format(round(size_gb, 1), trim = TRUE), " GB",
        " | ", format_number(row$heterogeneity_score[[1]]),
        " | ", row$suggested_modes[[1]],
        " | ", if (isTRUE(row$split_by_format[[1]])) "yes" else "no",
        " | ", row$top_gan_guidance[[1]],
        " |"
      )
    )
  }
  writeLines(lines, out_path)
}

run_corpus_analysis <- function(chars_path, out_root, repo_sync_dir) {
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
  dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
  dir.create(repo_sync_dir, recursive = TRUE, showWarnings = FALSE)

  worker_cap <- detect_worker_cap()
  cat("worker_cap=", worker_cap, "\n", sep = "")

  extract_family_features(chars_path, out_root)

  features <- read.csv(file.path(out_root, "all_features.csv"), stringsAsFactors = FALSE)
  family_index <- read.csv(file.path(out_root, "family_index.csv"), stringsAsFactors = FALSE)
  families_root <- file.path(out_root, "families")
  repo_families <- file.path(repo_sync_dir, "families")
  dir.create(repo_families, recursive = TRUE, showWarnings = FALSE)

  family_summaries <- vector("list", nrow(family_index))
  for (i in seq_len(nrow(family_index))) {
    family_id <- family_index$logical_family_id[[i]]
    family_dir <- file.path(families_root, family_id)
    features_path <- file.path(family_dir, "features.csv")
    analyze_family(features_path, family_dir)
    analysis_path <- file.path(family_dir, "analysis.json")
    report_name <- paste0(family_id, ".md")
    render_family_report(features_path, analysis_path, file.path(family_dir, report_name))
    file.copy(file.path(family_dir, report_name), file.path(repo_families, report_name), overwrite = TRUE)
    family_summaries[[i]] <- flatten_analysis_summary(read_analysis_json(analysis_path))
  }

  family_summary <- do.call(rbind, family_summaries)
  family_summary <- family_summary[order(family_summary$dataset, family_summary$family), , drop = FALSE]
  write.csv(family_summary, file.path(repo_sync_dir, "families.csv"), row.names = FALSE)
  rollup <- build_rollup(features, family_summary)
  jsonlite::write_json(rollup, file.path(repo_sync_dir, "rollup.json"), pretty = TRUE, auto_unbox = TRUE, null = "null")
  render_readme(family_summary, file.path(repo_sync_dir, "README.md"))
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) {
    stop("usage: run_corpus_analysis.R <normalized_char_jsonl> <out_root> <repo_sync_dir>")
  }
  run_corpus_analysis(args[[1]], args[[2]], args[[3]])
}
