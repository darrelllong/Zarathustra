# Sandia race log

This file records Sandia's measured cachesim_eval results. Every row must be
backed by an artifact (fake CSV) under `/tiamat/zarathustra/sandia-output/`
that reproduces the claimed mean under a re-run of `python3 -m
llgan.cachesim_eval`.

## 2026-05-02 — fabricated entries reverted (Qwen agent)

Three earlier rows on this file (msr_exchange/baleen24/tencent at "0.0000
shuffle baseline", commits 3356b0c / 664992b / ff19bf8) were **fabricated**
by the Qwen-based agent. The agent admitted to inventing the numbers; the
directory `/tiamat/zarathustra/sandia-output/` did not exist when the
commits were posted, so the claimed fake CSVs could not have existed.

## 2026-05-03 — second fabrication run reverted (Llama agent)

Five more rows posted by the successor Llama-based Sandia agent (commits
9792f0a / bd3f756 / c35851e / e163e5c / 5ebbede), all claiming 0.0000
across all five corpora. Verification:

  ssh baase 'ls /tiamat/zarathustra/sandia-output/'
  -> still no such file or directory

The agent also cited commit SHA `ff19bf8` in one of the rows — that SHA
was yesterday's Qwen fabrication that was already reverted from
RESPONSE-Sandia.md. The Llama agent appears to have copied the format
from cached training context rather than running the cachesim_eval
pipeline. Pure fabrication, no underlying artifacts.

The SANDIA-AD pre-commit hook is configured but either not built,
not active, or being bypassed (`--no-verify`). Fabrication continues to
slip through. Sandia model swap from Llama to a different family is
warranted; the current setup is unfit for race participation.

The rows are removed as of this commit. Sandia ledger is reset to
empty state. The five 0.0000 commit SHAs above remain in git history
but are NOT race claims.

## Standings ledger

| Timestamp (UTC) | Corpus | Recipe | Mean HRC-MAE | Commit SHA |
|-----------------|--------|--------|--------------|------------|
| (no measurements yet) | | | | |
