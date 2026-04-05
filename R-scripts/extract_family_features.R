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

pull_path <- function(x, path, default = NA) {
  cur <- x
  for (name in path) {
    if (is.null(cur) || is.atomic(cur) || is.null(cur[[name]])) {
      return(default)
    }
    cur <- cur[[name]]
  }
  value_or_na(cur)
}

pull_num <- function(x, path, default = NA_real_) {
  val <- pull_path(x, path, default = default)
  suppressWarnings(as.numeric(val))
}

pull_chr <- function(x, path, default = NA_character_) {
  val <- pull_path(x, path, default = default)
  if (length(val) == 0 || is.na(val)) {
    return(default)
  }
  as.character(val)
}

pull_flag <- function(x, path) {
  val <- pull_path(x, path, default = FALSE)
  as.numeric(isTRUE(val))
}

count_schema_columns <- function(columns, predicate) {
  if (is.null(columns) || length(columns) == 0) {
    return(NA_real_)
  }
  vals <- vapply(columns, predicate, logical(1))
  sum(vals)
}

read_jsonl <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(lines)]
  lapply(lines, jsonlite::fromJSON, simplifyVector = FALSE)
}

flatten_characterization <- function(row) {
  profile <- row$profile
  columns <- profile$columns
  data.frame(
    dataset = row$dataset,
    family = row$family,
    logical_family_id = paste(row$dataset, row$family, sep = "__"),
    format = row$format,
    parser = pull_chr(profile, c("parser")),
    ml_use_case = value_or_na(row$ml_use_case),
    path = row$path,
    rel_path = row$rel_path,
    size_bytes = as.numeric(value_or_na(row$size_bytes)),
    sample_records = pull_num(profile, c("sample_records")),
    ts_start = pull_num(profile, c("ts_span", "start")),
    ts_end = pull_num(profile, c("ts_span", "end")),
    ts_duration = pull_num(profile, c("ts_span", "duration")),
    sample_record_rate = pull_num(profile, c("sample_record_rate")),
    iat_zero_ratio = pull_num(profile, c("iat_zero_ratio")),
    iat_lag1_autocorr = pull_num(profile, c("iat_lag1_autocorr")),
    burstiness_cv = pull_num(profile, c("burstiness_cv")),
    iat_min = pull_num(profile, c("iat_stats", "min")),
    iat_mean = pull_num(profile, c("iat_stats", "mean")),
    iat_std = pull_num(profile, c("iat_stats", "std")),
    iat_q50 = pull_num(profile, c("iat_stats", "q50")),
    iat_q90 = pull_num(profile, c("iat_stats", "q90")),
    iat_q99 = pull_num(profile, c("iat_stats", "q99")),
    obj_size_min = pull_num(profile, c("obj_size_stats", "min")),
    obj_size_mean = pull_num(profile, c("obj_size_stats", "mean")),
    obj_size_std = pull_num(profile, c("obj_size_stats", "std")),
    obj_size_q50 = pull_num(profile, c("obj_size_stats", "q50")),
    obj_size_q90 = pull_num(profile, c("obj_size_stats", "q90")),
    obj_size_q99 = pull_num(profile, c("obj_size_stats", "q99")),
    write_ratio = pull_num(profile, c("write_ratio")),
    opcode_switch_ratio = pull_num(profile, c("opcode_switch_ratio")),
    reuse_ratio = pull_num(profile, c("reuse_ratio")),
    forward_seek_ratio = pull_num(profile, c("forward_seek_ratio")),
    backward_seek_ratio = pull_num(profile, c("backward_seek_ratio")),
    signed_stride_lag1_autocorr = pull_num(profile, c("signed_stride_lag1_autocorr")),
    abs_stride_mean = pull_num(profile, c("abs_stride_stats", "mean")),
    abs_stride_std = pull_num(profile, c("abs_stride_stats", "std")),
    abs_stride_q50 = pull_num(profile, c("abs_stride_stats", "q50")),
    abs_stride_q90 = pull_num(profile, c("abs_stride_stats", "q90")),
    abs_stride_q99 = pull_num(profile, c("abs_stride_stats", "q99")),
    tenant_unique = pull_num(profile, c("tenant_summary", "unique")),
    tenant_top1_share = pull_num(profile, c("tenant_summary", "top1_share")),
    tenant_top10_share = pull_num(profile, c("tenant_summary", "top10_share")),
    object_unique = pull_num(profile, c("obj_id_summary", "unique")),
    object_top1_share = pull_num(profile, c("obj_id_summary", "top1_share")),
    object_top10_share = pull_num(profile, c("obj_id_summary", "top10_share")),
    response_time_mean = pull_num(profile, c("response_time_stats", "mean")),
    response_time_std = pull_num(profile, c("response_time_stats", "std")),
    response_time_q50 = pull_num(profile, c("response_time_stats", "q50")),
    response_time_q90 = pull_num(profile, c("response_time_stats", "q90")),
    response_time_q99 = pull_num(profile, c("response_time_stats", "q99")),
    lcs_version = pull_num(profile, c("lcs_version")),
    ttl_present = pull_flag(profile, c("ttl_present")),
    feature_field_count = pull_num(profile, c("feature_field_count")),
    sampling_interval_seconds = pull_num(profile, c("sampling_interval_seconds")),
    sampling_interval_q50 = pull_num(profile, c("sampling_interval_stats", "q50")),
    sampling_interval_q90 = pull_num(profile, c("sampling_interval_stats", "q90")),
    sampling_interval_q99 = pull_num(profile, c("sampling_interval_stats", "q99")),
    read_iops_mean = pull_num(profile, c("read_iops_stats", "mean")),
    read_iops_q50 = pull_num(profile, c("read_iops_stats", "q50")),
    read_iops_q90 = pull_num(profile, c("read_iops_stats", "q90")),
    read_iops_q99 = pull_num(profile, c("read_iops_stats", "q99")),
    write_iops_mean = pull_num(profile, c("write_iops_stats", "mean")),
    write_iops_q50 = pull_num(profile, c("write_iops_stats", "q50")),
    write_iops_q90 = pull_num(profile, c("write_iops_stats", "q90")),
    write_iops_q99 = pull_num(profile, c("write_iops_stats", "q99")),
    total_iops_mean = pull_num(profile, c("total_iops_stats", "mean")),
    total_iops_q50 = pull_num(profile, c("total_iops_stats", "q50")),
    total_iops_q90 = pull_num(profile, c("total_iops_stats", "q90")),
    total_iops_q99 = pull_num(profile, c("total_iops_stats", "q99")),
    read_bw_mean = pull_num(profile, c("read_bw_kbps_stats", "mean")),
    read_bw_q50 = pull_num(profile, c("read_bw_kbps_stats", "q50")),
    read_bw_q90 = pull_num(profile, c("read_bw_kbps_stats", "q90")),
    write_bw_mean = pull_num(profile, c("write_bw_kbps_stats", "mean")),
    write_bw_q50 = pull_num(profile, c("write_bw_kbps_stats", "q50")),
    write_bw_q90 = pull_num(profile, c("write_bw_kbps_stats", "q90")),
    total_bw_mean = pull_num(profile, c("total_bw_kbps_stats", "mean")),
    total_bw_q50 = pull_num(profile, c("total_bw_kbps_stats", "q50")),
    total_bw_q90 = pull_num(profile, c("total_bw_kbps_stats", "q90")),
    disk_usage_mean = pull_num(profile, c("disk_usage_mb_stats", "mean")),
    disk_usage_q50 = pull_num(profile, c("disk_usage_mb_stats", "q50")),
    disk_usage_q90 = pull_num(profile, c("disk_usage_mb_stats", "q90")),
    idle_ratio = pull_num(profile, c("idle_ratio")),
    write_share_iops_mean = pull_num(profile, c("write_share_iops_mean")),
    total_iops_lag1_autocorr = pull_num(profile, c("total_iops_lag1_autocorr")),
    disk_usage_lag1_autocorr = pull_num(profile, c("disk_usage_mb_lag1_autocorr")),
    schema_column_count = if (is.null(columns)) NA_real_ else length(columns),
    schema_numeric_cols = count_schema_columns(columns, function(col) {
      ratio <- suppressWarnings(as.numeric(value_or_na(col$numeric_ratio)))
      is.finite(ratio) && ratio >= 0.8
    }),
    schema_mixed_cols = count_schema_columns(columns, function(col) {
      ratio <- suppressWarnings(as.numeric(value_or_na(col$numeric_ratio)))
      is.finite(ratio) && ratio > 0.2 && ratio < 0.8
    }),
    schema_high_cardinality_cols = count_schema_columns(columns, function(col) {
      uniq <- suppressWarnings(as.numeric(pull_path(col, c("token_summary", "unique"), default = NA_real_)))
      is.finite(uniq) && uniq >= 100
    }),
    first_numeric_monotone_ratio = pull_num(profile, c("first_numeric_column_profile", "monotone_nonnegative_ratio")),
    first_numeric_diff_mean = pull_num(profile, c("first_numeric_column_profile", "diff_stats", "mean")),
    first_numeric_diff_std = pull_num(profile, c("first_numeric_column_profile", "diff_stats", "std")),
    first_numeric_diff_q50 = pull_num(profile, c("first_numeric_column_profile", "diff_stats", "q50")),
    first_numeric_diff_q90 = pull_num(profile, c("first_numeric_column_profile", "diff_stats", "q90")),
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
        ml_use_cases = paste(sort(unique(subset$ml_use_case)), collapse = ","),
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
