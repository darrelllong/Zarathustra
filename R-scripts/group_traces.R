#!/usr/bin/env Rscript
# Cross-trace grouping for Zarathustra. Joins existing all_features.csv
# with the new new_features.csv, scales, and runs:
#   * UMAP 2D projection
#   * k-medoids (PAM) with k = 2..10
#   * silhouette + gap statistic
#   * hierarchical clustering (ward.D2) for dendrogram
# Emits trace_clusters.csv (per-trace cluster ids, UMAP coords) and
# family_cohesion.csv (within-family cluster purity).

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(cluster)
})

args <- commandArgs(trailingOnly = TRUE)
roots <- c("/tiamat", "/Volumes/Gigantor")
root <- roots[which(file.exists(roots))[1]]
default_existing <- file.path(root, "zarathustra", "r-output",
                              "full_run_20260430", "all_features.csv")
default_new <- file.path(root, "zarathustra", "r-output",
                         paste0("append_run_", format(Sys.Date(), "%Y%m%d")),
                         "new_features.csv")
default_out <- dirname(default_new)
existing_path <- if (length(args) >= 1) args[[1]] else default_existing
new_path <- if (length(args) >= 2) args[[2]] else default_new
out_dir <- if (length(args) >= 3) args[[3]] else default_out

user_lib <- Sys.getenv("R_LIBS_USER", unset = "~/R/library")
.libPaths(c(path.expand(user_lib), .libPaths()))

cat(sprintf("[group] existing=%s\n[group] new=%s\n[group] out=%s\n",
            existing_path, new_path, out_dir))

existing <- if (file.exists(existing_path)) fread(existing_path) else NULL
new_feats <- fread(new_path)

# Per-trace key for joining: prefer (path) since paths are unique.
if (!is.null(existing) && "path" %in% names(existing)) {
  joined <- merge(new_feats, existing, by = "path", all.x = TRUE,
                  suffixes = c("", ".existing"))
} else {
  joined <- new_feats
}
cat(sprintf("[group] joined %d traces, %d numeric features\n",
            nrow(joined),
            sum(vapply(joined, is.numeric, logical(1L)))))

# Build feature matrix: only numeric, drop columns with >50% NA, drop
# zero-variance, log1p any column with min >= 0 and max/min ratio > 1000.
num_cols <- names(joined)[vapply(joined, is.numeric, logical(1L))]
M <- as.matrix(joined[, ..num_cols])
na_share <- colMeans(is.na(M))
M <- M[, na_share < 0.5, drop = FALSE]
v <- apply(M, 2L, function(x) var(x, na.rm = TRUE))
M <- M[, is.finite(v) & v > 0, drop = FALSE]
mn <- apply(M, 2L, min, na.rm = TRUE)
mx <- apply(M, 2L, max, na.rm = TRUE)
log_cols <- which(mn >= 0 & is.finite(mx) & mx > 0 & mx / pmax(mn, 1e-12) > 1000)
for (i in log_cols) M[, i] <- log1p(M[, i])
# Median-impute remaining NAs, then scale.
for (j in seq_len(ncol(M))) {
  na_idx <- which(is.na(M[, j]) | !is.finite(M[, j]))
  if (length(na_idx)) M[na_idx, j] <- median(M[, j], na.rm = TRUE)
}
M <- M[, apply(M, 2L, function(x) length(unique(x)) > 1L), drop = FALSE]
fz <- scale(M)
fz[!is.finite(fz)] <- 0
cat(sprintf("[group] feature matrix: %d x %d\n", nrow(fz), ncol(fz)))

# UMAP
umap_xy <- NULL
if (requireNamespace("uwot", quietly = TRUE) && nrow(fz) >= 5) {
  umap_xy <- try(uwot::umap(fz, n_neighbors = min(15L, nrow(fz) - 1L),
                            min_dist = 0.1, n_components = 2L,
                            verbose = FALSE), silent = TRUE)
  if (inherits(umap_xy, "try-error")) umap_xy <- NULL
}

# k-medoids with PAM, k=2..min(10, n-1)
ks <- 2:min(10L, nrow(fz) - 1L)
sil_pam <- numeric(length(ks))
clus_list <- vector("list", length(ks))
for (i in seq_along(ks)) {
  k <- ks[i]
  res <- try(cluster::pam(fz, k = k, metric = "euclidean", stand = FALSE),
             silent = TRUE)
  if (inherits(res, "try-error")) {
    sil_pam[i] <- NA_real_; clus_list[[i]] <- rep(NA_integer_, nrow(fz))
  } else {
    sil_pam[i] <- res$silinfo$avg.width
    clus_list[[i]] <- res$clustering
  }
}
best_k_idx <- which.max(sil_pam)
best_k <- ks[best_k_idx]
best_clus <- clus_list[[best_k_idx]]

# Hierarchical
hc <- hclust(dist(fz), method = "ward.D2")
hc_cuts <- cutree(hc, k = best_k)

# Gap statistic for kmedoids — using the full sample is expensive; cap at 200 rows.
gap_dt <- NULL
if (nrow(fz) >= 8) {
  gap_n <- min(200L, nrow(fz))
  gap_idx <- sort(sample.int(nrow(fz), gap_n))
  pamFun <- function(x, k) list(cluster = cluster::pam(x, k)$clustering)
  gap <- try(cluster::clusGap(fz[gap_idx, , drop = FALSE], FUN = pamFun,
                              K.max = min(8L, gap_n - 1L), B = 50L),
             silent = TRUE)
  if (!inherits(gap, "try-error")) {
    gap_dt <- data.table(k = seq_along(gap$Tab[, "gap"]),
                         gap = gap$Tab[, "gap"],
                         SE_sim = gap$Tab[, "SE.sim"])
  }
}

clusters <- data.table(
  path = joined$path,
  dataset = joined$dataset,
  family = joined$family,
  logical_family_id = joined$logical_family_id,
  pam_cluster = best_clus,
  hclust_cluster = hc_cuts
)
if (!is.null(umap_xy)) {
  clusters[, umap1 := umap_xy[, 1]]
  clusters[, umap2 := umap_xy[, 2]]
}
fwrite(clusters, file.path(out_dir, "trace_clusters.csv"))
fwrite(data.table(k = ks, pam_silhouette = sil_pam),
       file.path(out_dir, "cluster_diagnostics.csv"))
if (!is.null(gap_dt)) fwrite(gap_dt, file.path(out_dir, "cluster_gap_statistic.csv"))

# Family cohesion: dominant cluster share per family
cohesion <- clusters[, {
  tab <- table(pam_cluster)
  list(n = .N,
       dominant_cluster = as.integer(names(tab)[which.max(tab)]),
       dominant_share = max(tab) / .N,
       n_clusters_seen = length(tab))
}, by = .(dataset, family)]
setorder(cohesion, -dominant_share)
fwrite(cohesion, file.path(out_dir, "family_cohesion.csv"))

cat(sprintf("[group] best_k=%d (silhouette=%.4f)\n", best_k, sil_pam[best_k_idx]))
cat(sprintf("[group] wrote %s/trace_clusters.csv\n", out_dir))
