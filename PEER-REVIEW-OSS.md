# Peer‑Review OSS Process

This document outlines a lightweight, reproducible peer‑review workflow for the **Zarathustra** project.  It replaces the fragmented `PEER‑REVIEW*.md` files that currently live in the repo.

## 1. Review workflow

* **Submit a PR** – All changes that touch implementation, configuration or data‑generation should be committed via a pull request.
* **Label** – The PR must have the label `review‑needed` added automatically by the PR template.
* **Reviewers** – Two reviewers are requested: one from the maintainers list and, if possible, an external contributor.  Reviewer comments should be added directly to the PR comment thread.
* **Checklist** – Before merging:
  1. Unit tests pass on the base branch.
  2. Code style passes `black`, `ruff`, and `isort`.
  3. Documentation changes (README, module docstrings) are complete.
  4. If the change affects experimental results, a brief summary of the impact (e.g., “improved MMD² from 0.011 to 0.009”) must be added.
  5. Peer‑review action items are moved from the `TODO.md` file or the `PEER‑REVIEW-OSS.md` body so that they are resolved before merge.

## 2. Action Items

| Item | Owner | Status |
|------|-------|--------|
| Move shared utilities to `zarathustra/` | darrell | ✅ |
| Add tests for model forward pass | TBD | ☐ |
| Implement CI workflow (tests + lint) | TBD | ☐ |
| Update CONTRIBUTING.md | darrell | ✅ |
| Merge all `PEER‑REVIEW*.md` into this file | darrell | ✅ |
| Close old review comments | All | ✅ |

## 3. Documentation strategy

* Each sub‑module (`llgan`, `altgan`) gets a short README that links to the relevant usage examples.
* A high‑level `README.md` references the project structure and where to find the best‑running configurations.
* The `CHANGELOG.md` is auto‑generated from commit messages that follow the conventional‑commit format.

## 4. Reproducibility

* **Seed** – All training scripts accept `--seed` and the default seed is documented.
* **Datasets** – The dataset locations are stored in `CONFIG.md` and can be downloaded via the `scripts/download_data.sh` helper.
* **Environment** – The repository ships a `requirements.txt` that is used by the CI pipeline to create a virtual environment.  For local reproducibility a Conda environment file (`environment.yml`) is also provided.

---
**Please** keep all future updates to the peer‑review process in this file.
Thank you!