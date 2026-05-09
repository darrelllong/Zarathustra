#!/usr/bin/env Rscript
# Extract extended features for the 7 2017_docker parquet traces.
# Calls dump_parquet_docker.py via the venv's python, then runs the same
# analyze_series logic from extract_extended_features.R.

suppressPackageStartupMessages({
  library(data.table); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else stop("usage: OUT_DIR")
max_records <- if (length(args) >= 2) as.integer(args[[2]]) else 1000000L
venv_py <- if (length(args) >= 3) args[[3]] else "/tmp/parquet-env/bin/python3"
dump_py <- if (length(args) >= 4) args[[4]] else
  file.path("/tiamat/zarathustra/r-analysis-src/R-scripts/dump_parquet_docker.py")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
per_trace_dir <- file.path(out_dir, "per_trace"); dir.create(per_trace_dir, showWarnings = FALSE)

user_lib <- Sys.getenv("R_LIBS_USER", unset = "~/R/library")
.libPaths(c(path.expand(user_lib), .libPaths()))

# Source the analysis primitives by inlining the slim extractor's analyze_series
# (we just need analyze_series + helpers — not its main loop).
source("/tiamat/zarathustra/r-analysis-src/R-scripts/extract_extended_features.R", echo = FALSE)
