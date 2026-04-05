#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
user_lib <- Sys.getenv("R_LIBS_USER", unset = "~/R/library")
repos <- "https://cloud.r-project.org"
packages <- c(
  "jsonlite",
  "readr",
  "dplyr",
  "tidyr",
  "stringr",
  "ggplot2",
  "patchwork",
  "corrplot",
  "moments",
  "e1071",
  "changepoint",
  "strucchange",
  "forecast",
  "tsfeatures",
  "dbscan",
  "mclust",
  "factoextra"
)

dir.create(path.expand(user_lib), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(path.expand(user_lib), .libPaths()))

missing <- packages[!vapply(packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0) {
  install.packages(missing, lib = path.expand(user_lib), repos = repos, Ncpus = 2L)
}

cat("R_LIBS_USER=", path.expand(user_lib), "\n", sep = "")
cat("Installed packages:", paste(packages, collapse = ", "), "\n")
