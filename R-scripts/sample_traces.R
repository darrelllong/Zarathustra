#!/usr/bin/env Rscript
# Pick up to N representative .zst traces per logical family using the
# FAMILY_REGISTRY in parsers/core.py. Stratified by file-size decile,
# deterministic seed = digest(family_id). Emits sample_manifest.csv.
#
# Output: <out_dir>/sample_manifest.csv with columns:
#   dataset, family, logical_family_id, format, kind, path, size_bytes, decile

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(digest)
})

# --- Path detection (vinge: /tiamat ; gigantor: /Volumes/Gigantor) ----------
roots <- c("/tiamat", "/Volumes/Gigantor")
root <- roots[which(file.exists(roots))[1]]
if (is.na(root)) stop("Could not find /tiamat or /Volumes/Gigantor")
traces_root <- file.path(root, "zarathustra", "traces")
default_out <- file.path(root, "zarathustra", "r-output",
                         paste0("append_run_", format(Sys.Date(), "%Y%m%d")))

# --- Args -------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
n_per_family <- if (length(args) >= 1) as.integer(args[[1]]) else 8L
out_dir <- if (length(args) >= 2) args[[2]] else default_out
parsers_py <- if (length(args) >= 3) args[[3]] else
              file.path(Sys.getenv("HOME"), "LLNL", "Zarathustra", "parsers",
                        "core.py")
if (!file.exists(parsers_py)) {
  alt <- file.path(root, "zarathustra", "r-analysis-src", "parsers", "core.py")
  if (file.exists(alt)) parsers_py <- alt
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("[sample] root=%s  N=%d  out=%s\n", root, n_per_family, out_dir))
cat(sprintf("[sample] parsers=%s\n", parsers_py))

# --- Resolve identities via core.py canonical_identity_for_path -----------
# Enumerate .zst traces first.
all_zst <- list.files(traces_root, pattern = "\\.zst$", recursive = TRUE,
                      full.names = TRUE)
cat(sprintf("[sample] %d .zst files under %s\n", length(all_zst), traces_root))
paths_file <- tempfile(fileext = ".txt")
writeLines(all_zst, paths_file)

py_helper <- tempfile(fileext = ".py")
writeLines(c(
  "import json, sys, importlib.util",
  sprintf("spec = importlib.util.spec_from_file_location('zarathustra_core', %s)",
          shQuote(parsers_py)),
  "m = importlib.util.module_from_spec(spec)",
  "sys.modules['zarathustra_core'] = m",
  "spec.loader.exec_module(m)",
  "registry = []",
  "for (ds, fam), meta in m.FAMILY_REGISTRY.items():",
  "    registry.append({'dataset': ds, 'family': fam,",
  "                     'kind': meta.get('kind'),",
  "                     'formats': list(meta.get('formats', []))})",
  sprintf("paths = [l.strip() for l in open(%s) if l.strip()]",
          shQuote(paths_file)),
  "rows = []",
  "for p in paths:",
  "    try:",
  "        ident = m.canonical_identity_for_path(p)",
  "        rows.append({'path': p, 'dataset': ident.dataset,",
  "                     'family': ident.family, 'format': ident.format,",
  "                     'logical_family_id': ident.logical_family_id})",
  "    except Exception as e:",
  "        rows.append({'path': p, 'error': str(e)})",
  "json.dump({'registry': registry, 'rows': rows}, sys.stdout)"
), py_helper)
ident_json <- system2("python3", args = py_helper, stdout = TRUE)
ident <- fromJSON(paste(ident_json, collapse = ""), simplifyDataFrame = TRUE)
registry <- as.data.table(ident$registry)
all_dt <- as.data.table(ident$rows)
cat(sprintf("[sample] %d logical families in registry\n", nrow(registry)))
cat(sprintf("[sample] resolved identities for %d / %d paths\n",
            sum(!is.na(all_dt$dataset)), nrow(all_dt)))
all_dt <- all_dt[!is.na(dataset)]
all_dt[, size_bytes := file.info(path)$size]

# --- Sample per registered family ------------------------------------------
sample_one_family <- function(rows, n) {
  if (nrow(rows) == 0) return(rows[0])
  if (nrow(rows) <= n) return(rows)
  # Stratify by size decile; pick proportionally with deterministic seed.
  decile <- cut(rows$size_bytes,
                breaks = unique(quantile(rows$size_bytes,
                                         probs = seq(0, 1, by = 0.1),
                                         type = 7, na.rm = TRUE)),
                include.lowest = TRUE, labels = FALSE)
  rows[, decile := decile]
  seed_int <- strtoi(substr(digest(rows$logical_family_id[[1]],
                                   algo = "sha1"), 1, 8), base = 16L)
  if (is.na(seed_int)) seed_int <- 1L
  set.seed(seed_int)
  # First take one per decile, then top up to n at random.
  per_decile <- rows[, .SD[sample(.N, 1L)], by = decile]
  remaining <- rows[!path %in% per_decile$path]
  need <- n - nrow(per_decile)
  if (need > 0 && nrow(remaining) > 0) {
    extra <- remaining[sample(.N, min(need, .N))]
    rbind(per_decile, extra, fill = TRUE)
  } else {
    per_decile[seq_len(n)]
  }
}

manifest_rows <- list()
for (i in seq_len(nrow(registry))) {
  ds <- registry$dataset[i]
  fam <- registry$family[i]
  rows <- all_dt[dataset == ds & family == fam]
  if (nrow(rows) == 0) next
  picked <- sample_one_family(rows, n_per_family)
  picked[, kind := registry$kind[i]]
  picked[, registered_formats := paste(registry$formats[[i]], collapse = "|")]
  manifest_rows[[length(manifest_rows) + 1L]] <- picked
}
manifest <- rbindlist(manifest_rows, use.names = TRUE, fill = TRUE)
setorder(manifest, dataset, family, size_bytes)

# --- Surface unmatched dataset/family directories so we don't silently miss --
matched_keys <- unique(paste(manifest$dataset, manifest$family, sep = "/"))
all_keys <- unique(paste(all_dt$dataset, all_dt$family, sep = "/"))
unmatched <- setdiff(all_keys, matched_keys)
if (length(unmatched) > 0) {
  cat("[sample] WARNING: unmatched dataset/family directories (not in registry):\n")
  cat(paste0("  ", unmatched, collapse = "\n"), "\n", sep = "")
}

manifest_path <- file.path(out_dir, "sample_manifest.csv")
fwrite(manifest, manifest_path)
cat(sprintf("[sample] wrote %d rows to %s\n", nrow(manifest), manifest_path))
cat(sprintf("[sample] families covered: %d / %d\n",
            length(unique(manifest$logical_family_id)), nrow(registry)))
