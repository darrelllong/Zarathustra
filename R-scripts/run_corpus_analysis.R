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

build_rollup <- function(features, family_index) {
  groups <- list()
  for (i in seq_len(nrow(family_index))) {
    family_id <- family_index$logical_family_id[[i]]
    subset <- features[features$logical_family_id == family_id, , drop = FALSE]
    groups[[family_id]] <- list(
      dataset = subset$dataset[[1]],
      family = subset$family[[1]],
      files = nrow(subset),
      bytes = sum(subset$size_bytes, na.rm = TRUE),
      formats = sort(unique(subset$format)),
      parsers = sort(unique(subset$parser)),
      write_ratio = median(subset$write_ratio, na.rm = TRUE),
      reuse_ratio = median(subset$reuse_ratio, na.rm = TRUE),
      burstiness_cv = median(subset$burstiness_cv, na.rm = TRUE),
      iat_q50 = median(subset$iat_q50, na.rm = TRUE),
      obj_size_q50 = median(subset$obj_size_q50, na.rm = TRUE),
      tenant_unique = median(subset$tenant_unique, na.rm = TRUE)
    )
  }
  groups
}

render_readme <- function(family_index, features, out_path) {
  lines <- c(
    "# Trace Family Characterizations",
    "",
    paste0("**Total logical families:** ", nrow(family_index), "  "),
    paste0("**Total trace files summarized:** ", nrow(features), "  "),
    "",
    "| Family | Dataset | Files | Size | write_ratio | reuse_ratio | burstiness_cv | iat_q50 |",
    "|---|---|---:|---:|---:|---:|---:|---:|"
  )
  for (i in seq_len(nrow(family_index))) {
    subset <- features[features$logical_family_id == family_index$logical_family_id[[i]], , drop = FALSE]
    size_gb <- sum(subset$size_bytes, na.rm = TRUE) / (1024 ^ 3)
    lines <- c(
      lines,
      paste0(
        "| ", family_index$family[[i]],
        " | ", family_index$dataset[[i]],
        " | ", nrow(subset),
        " | ", format(round(size_gb, 1), trim = TRUE), " GB",
        " | ", format_number(median(subset$write_ratio, na.rm = TRUE)),
        " | ", format_number(median(subset$reuse_ratio, na.rm = TRUE)),
        " | ", format_number(median(subset$burstiness_cv, na.rm = TRUE)),
        " | ", format_number(median(subset$iat_q50, na.rm = TRUE)),
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

  for (i in seq_len(nrow(family_index))) {
    family_id <- family_index$logical_family_id[[i]]
    family_dir <- file.path(families_root, family_id)
    features_path <- file.path(family_dir, "features.csv")
    analyze_family(features_path, family_dir)
    report_name <- paste0(family_id, ".md")
    render_family_report(features_path, file.path(family_dir, "analysis.json"), file.path(family_dir, report_name))
    file.copy(file.path(family_dir, report_name), file.path(repo_families, report_name), overwrite = TRUE)
  }

  file.copy(file.path(out_root, "family_index.csv"), file.path(repo_sync_dir, "families.csv"), overwrite = TRUE)
  rollup <- build_rollup(features, family_index)
  jsonlite::write_json(rollup, file.path(repo_sync_dir, "rollup.json"), pretty = TRUE, auto_unbox = TRUE, null = "null")
  render_readme(family_index, features, file.path(repo_sync_dir, "README.md"))
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) {
    stop("usage: run_corpus_analysis.R <normalized_char_jsonl> <out_root> <repo_sync_dir>")
  }
  run_corpus_analysis(args[[1]], args[[2]], args[[3]])
}
