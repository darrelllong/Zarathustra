#!/usr/bin/env Rscript

suppressWarnings({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required; run install_packages.R first")
  }
})

value_or_na <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(NA)
  }
  if (is.list(x) && length(x) == 0) {
    return(NA)
  }
  x
}

read_jsonl <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(lines)]
  lapply(lines, jsonlite::fromJSON, simplifyVector = FALSE)
}

flatten_characterization <- function(row) {
  profile <- row$profile
  data.frame(
    dataset = row$dataset,
    family = row$family,
    logical_family_id = paste(row$dataset, row$family, sep = "__"),
    format = row$format,
    parser = value_or_na(profile$parser),
    path = row$path,
    rel_path = row$rel_path,
    size_bytes = as.numeric(value_or_na(row$size_bytes)),
    sample_records = as.numeric(value_or_na(profile$sample_records)),
    write_ratio = as.numeric(value_or_na(profile$write_ratio)),
    reuse_ratio = as.numeric(value_or_na(profile$reuse_ratio)),
    burstiness_cv = as.numeric(value_or_na(profile$burstiness_cv)),
    iat_q50 = as.numeric(value_or_na(profile$iat_stats$q50)),
    iat_q90 = as.numeric(value_or_na(profile$iat_stats$q90)),
    obj_size_q50 = as.numeric(value_or_na(profile$obj_size_stats$q50)),
    obj_size_q90 = as.numeric(value_or_na(profile$obj_size_stats$q90)),
    tenant_unique = as.numeric(value_or_na(profile$tenant_summary$unique)),
    idle_ratio = as.numeric(value_or_na(profile$idle_ratio)),
    total_iops_q50 = as.numeric(value_or_na(profile$total_iops_stats$q50)),
    opcode_switch_ratio = as.numeric(value_or_na(profile$opcode_switch_ratio)),
    iat_lag1_autocorr = as.numeric(value_or_na(profile$iat_lag1_autocorr)),
    forward_seek_ratio = as.numeric(value_or_na(profile$forward_seek_ratio)),
    backward_seek_ratio = as.numeric(value_or_na(profile$backward_seek_ratio)),
    stringsAsFactors = FALSE
  )
}

extract_family_features <- function(chars_path, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  rows <- read_jsonl(chars_path)
  flat_rows <- lapply(rows, flatten_characterization)
  features <- do.call(rbind, flat_rows)
  features <- features[order(features$dataset, features$family, features$format, features$rel_path), ]

  write.csv(features, file.path(out_dir, "all_features.csv"), row.names = FALSE)

  logical_ids <- unique(features$logical_family_id)
  family_index <- do.call(
    rbind,
    lapply(logical_ids, function(family_id) {
      subset <- features[features$logical_family_id == family_id, ]
      data.frame(
        logical_family_id = family_id,
        dataset = subset$dataset[[1]],
        family = subset$family[[1]],
        files = nrow(subset),
        bytes = sum(subset$size_bytes, na.rm = TRUE),
        formats = paste(sort(unique(subset$format)), collapse = ","),
        parsers = paste(sort(unique(subset$parser)), collapse = ","),
        stringsAsFactors = FALSE
      )
    })
  )
  family_index <- family_index[order(family_index$dataset, family_index$family), ]
  write.csv(family_index, file.path(out_dir, "family_index.csv"), row.names = FALSE)

  families_dir <- file.path(out_dir, "families")
  dir.create(families_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_len(nrow(family_index))) {
    family_id <- family_index$logical_family_id[[i]]
    subset <- features[features$logical_family_id == family_id, ]
    family_dir <- file.path(families_dir, family_id)
    dir.create(family_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(subset, file.path(family_dir, "features.csv"), row.names = FALSE)
  }

  invisible(features)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("usage: extract_family_features.R <trace_characterizations.normalized.jsonl> <out_dir>")
  }
  extract_family_features(args[[1]], args[[2]])
}
