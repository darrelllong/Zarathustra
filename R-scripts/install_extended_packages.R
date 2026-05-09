#!/usr/bin/env Rscript
user_lib <- Sys.getenv("R_LIBS_USER", unset = "~/R/library")
dir.create(path.expand(user_lib), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(path.expand(user_lib), .libPaths()))
repos <- "https://cloud.r-project.org"

required <- c(
  "lomb", "pracma", "nonlinearTseries", "wavelets", "WaveletComp",
  "evir", "eva", "entropy", "infotheo", "depmixS4", "uwot",
  "PerformanceAnalytics"
)
# fractal is archived on CRAN; pracma + nonlinearTseries give 5+ Hurst
# estimators between them. arrow drags in openssl + a long source build.
optional <- character()

install_set <- function(pkgs) {
  miss <- pkgs[!vapply(pkgs, requireNamespace, quietly = TRUE,
                       FUN.VALUE = logical(1))]
  if (length(miss) > 0) {
    install.packages(miss, lib = path.expand(user_lib), repos = repos,
                     Ncpus = 8L)
  }
}

install_set(required)
try(install_set(optional), silent = TRUE)

still_missing <- required[!vapply(required, requireNamespace, quietly = TRUE,
                                  FUN.VALUE = logical(1))]
cat("R_LIBS_USER=", path.expand(user_lib), "\n", sep = "")
opt_state <- vapply(optional, requireNamespace, quietly = TRUE,
                    FUN.VALUE = logical(1))
cat("optional packages: ",
    paste0(names(opt_state), "=", ifelse(opt_state, "ok", "missing"),
           collapse = ", "), "\n", sep = "")
if (length(still_missing) > 0) {
  cat("STILL MISSING (required): ",
      paste(still_missing, collapse = ", "), "\n", sep = "")
  quit(status = 1)
}
cat("All required extended packages installed.\n")
