# Peer Review

Round 2 below is the race-focused strategy review and is the one to use for deciding what to do next.

---

## Round 1

### Findings

1. `[P1]` The fallback PRDC implementation is incorrect, and it is active in the current environment. In [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L297), the code falls back when `prdc` is unavailable; locally, `python3 -c "import prdc"` fails with `ModuleNotFoundError`. In that fallback, [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L321) computes `r_fake` but never uses it, and both "recall" and "coverage" are derived from `in_real_ball.any(axis=1).mean()` at [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L330) and [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L337). That means the repo can currently emit invalid PRDC numbers from both `eval.py` and `compare.py`, which undermines a major part of the reported results.

2. `[P1]` `eval.py --baseline` can break on conditional checkpoints. The main fake-sampling path correctly passes real windows when conditioning is required, but the baseline path does not. `_sample_fake()` explicitly raises when `cond_dim > 0` and there is no `char_file` at [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L432), yet the baseline branch calls `_sample_fake(b_ckpt, n_samples, device)` unconditionally at [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L586). Any checkpoint comparison involving conditional models without serialized `char_file` metadata is therefore fragile or non-functional.

3. `[P1]` The reuse-rate metric is hard-coded to a feature index instead of the fitted schema, so reported locality results can be wrong. Evaluation assumes `obj_id_reuse` is always column 3 at [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L542) and [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L546), but training already knows column order is dynamic and resolves it from `prep.col_names` at [llgan/train.py](/Users/darrell/Zarathustra/llgan/train.py#L446). On corpora where opcode is retained, tenant is absent, or feature order changes after auto-drop, the current evaluation path can measure the wrong column and quietly corrupt the reuse-rate numbers in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L2080).

4. `[P2]` `TraceDataset` has an off-by-one bug that drops one valid window from every file and drops exact-length files entirely. The dataset length is set to `max(0, len(data) - timestep)` at [llgan/dataset.py](/Users/darrell/Zarathustra/llgan/dataset.py#L988), but `__getitem__` slices `idx:idx+timestep`, so the valid start indices are `0..len(data)-timestep`, inclusive. The correct count is `len(data) - timestep + 1`. As written, every file loses its final legal window, and any file with exactly `timestep` records contributes zero windows.

5. `[P2]` The validation MMD helpers can crash on small validation sets. Both `evaluate_mmd()` and `evaluate_metrics()` slice up to `n_samples` from the validation tensor, but then reshape using the requested sample count instead of the actual number returned: [llgan/mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py#L113), [llgan/mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py#L115), [llgan/mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py#L250), and [llgan/mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py#L252). If `len(val_data) < n_samples`, the code will fail instead of degrading gracefully. That is easy to hit on small-file experiments or narrow validation splits.

6. `[P2]` The characterization pipeline is tightly bound to one machine layout, which makes reproduction brittle. The parser root is hard-coded as `/tiamat/zarathustra/traces` at [parsers/core.py](/Users/darrell/Zarathustra/parsers/core.py#L19), and several analysis scripts also default to `/tiamat/zarathustra/...` paths, for example [traces/analysis/characterize_traces.py](/Users/darrell/Zarathustra/traces/analysis/characterize_traces.py#L25) and [traces/analysis/normalize_characterizations.py](/Users/darrell/Zarathustra/traces/analysis/normalize_characterizations.py#L97). That is workable for one host, but it means "results so far" are not cleanly reproducible from a fresh checkout without path surgery.

7. `[P2]` The repo is currently overstating the strength of the results. The README claims the system produces traces that are "statistically indistinguishable" at [README.md](/Users/darrell/Zarathustra/README.md#L11) and [README.md](/Users/darrell/Zarathustra/README.md#L13), and the paper draft makes the same claim at [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L65) and [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L214). But the repo's own metric notes say DMD-GEN is "stuck at ~0.71" at [llgan/mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py#L43), with repeated reminders in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1312) and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1420) that the target is `< 0.30`. Even the best full-eval combined score documented in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L2087) is framed as "all-time best," not as parity with real data. The safer claim today is "substantial improvement on fidelity/coverage metrics, with temporal dynamics still unresolved."

8. `[P3]` The paper is not yet in a reviewable-results state and still conflicts with the repo in a few places. It still contains unresolved placeholders for authors, hidden size, ablations, baselines, figures, and final results at [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L41), [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L467), [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L491), and [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L560). It also fixes the modeled feature dimension at `d = 5` at [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L211), while the code can train on 5 or 6 modeled columns depending on auto-drop and format, and its Tencent dataset count at [paper/main.tex](/Users/darrell/Zarathustra/paper/main.tex#L451) does not line up with the file counts described in [README.md](/Users/darrell/Zarathustra/README.md#L66) and [README.md](/Users/darrell/Zarathustra/README.md#L74). That does not block coding, but it does mean the written "results so far" are not publication-ready.

### Verification

- Read-through covered the main training, evaluation, generation, parser, trace-analysis, R-analysis, version-log, README, and paper paths.
- `python3 -m py_compile llgan/*.py parsers/*.py traces/analysis/*.py` passes, so the issues above are logic/result-quality problems rather than syntax breakage.
- The local environment does **not** have the `prdc` package installed, so the broken fallback path in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L297) is the one that would run here today.

### Short Take

The project has real momentum and a surprisingly strong amount of experiment hygiene in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md), but the current weak point is evaluation trustworthiness, not model ambition. Before leaning harder on new architectures, I would fix the metric-path bugs above, rerun the headline evaluations, and then soften the repo-level claims until the corrected numbers support them.

---

## Round 2

### Race-Focused Findings

1. `[P1]` The repo has already generated enough evidence that more scalar loss tuning is unlikely to win this race. [IDEAS.md](/Users/darrell/Zarathustra/IDEAS.md#L5) says that outright, and the version log backs it up: DMD-GEN is still around `0.70` and reuse is still around `0.005` even after repeated loss-family changes in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1312). My read is that the next breakthrough has to come from changing the structure of what the model is asked to represent, not adding one more weighted term.

2. `[P1]` The biggest architectural mismatch is that training is still mostly local-window while generation is long-horizon and stateful. The repo itself calls this out in [TODO.md](/Users/darrell/Zarathustra/TODO.md#L111): `generate.py` carries hidden state across windows, but training does not optimize a chunk-continuous objective. That lines up almost perfectly with the persistent temporal gap in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1312). If I had to bet on one move with the highest upside, it would be a stitched-window or chunk-continuity curriculum first, and then a hierarchical regime-to-event generator second, as already sketched in [TODO.md](/Users/darrell/Zarathustra/TODO.md#L293) and [IDEAS.md](/Users/darrell/Zarathustra/IDEAS.md#L79).

3. `[P1]` Multimodality needs to be explicit and corpus-specific, not "one clever setting for everything." The repo has already discovered that Alibaba and Tencent want different regime structures: K=2 clearly helped Alibaba in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L210), while K=2 clearly hurt Tencent in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L180). Likewise, char-file conditioning by itself made the model narrower rather than broader in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L842). The strongest implication is that the winning path is not a single universal recipe; it is two explicit playbooks:
   Tencent: strong multimodal prior or regime code with higher K.
   Alibaba: low-K regime separation with the extra conditioning noise that seems to help that corpus explore.

4. `[P1]` Pretraining is a competitive advantage and should be treated like one. The clearest sentence in the entire repo is probably [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1902): "pretrain quality is the dominant variable." That is a gift in a race. A 35-minute pretrain that moves the whole downstream trajectory is much cheaper than spending days on adversarial variants that start from a weaker latent space. I would operationalize this immediately: build a small bank of fresh pretrains, rank them, and only launch expensive Phase 3 runs from the best ones. Right now the repo treats pretrain quality as an observation; it should be a search axis.

5. `[P1]` Locality is still the Achilles' heel, and I do not think another scalar locality loss is enough. The repo keeps reporting reuse around `0.005` in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1314), while [TODO.md](/Users/darrell/Zarathustra/TODO.md#L154) already says locality should be first-class, not a side metric. My inference is that the next untried high-leverage idea is a domain-specific reuse mechanism: first decide reuse vs non-reuse, then either copy from a recent-object memory or generate a stride for a new seek. The current representation split (`obj_id_reuse` + `obj_id_stride`) is a good encoding, but the generator still behaves like a smooth regressor, not a system with an explicit copy path. If the goal is accurate synthetic I/O traces rather than just prettier metrics, this is where I would spend bold-idea budget.

6. `[P2]` Several shiny directions have already been empirically ruled out and should be treated as dead ends unless some new mechanism changes the premise. The evidence is pretty direct:
   2-layer LSTMs failed on both corpora in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L175).
   Projection critic made the critic trivially too strong in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L264).
   BayesGAN did not synergize with the regime sampler in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L256).
   Expanded conditioning hurt in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L110).
   Mixed-type recovery was not the bottleneck in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L382).
   The race advantage now comes from pruning, not just inventing.

7. `[P2]` Block sampling should stay corpus-specific rather than becoming doctrine. Tencent got some early benefit in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L154), but Alibaba got worse in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L89). That means the "respect temporal persistence" instinct is right, but the implementation is too blunt. I would keep it as a Tencent-only branch for now and not spend more Alibaba cycles on it.

8. `[P2]` HRC/cache-fidelity evaluation is not just for the paper; it can become a selection advantage. [IDEAS.md](/Users/darrell/Zarathustra/IDEAS.md#L228) makes the case well: low MMD does not guarantee realistic cache behavior. In a race, that matters because teams can overfit to the wrong proxy. I would not wait for paper cleanup here. Start using long-trace HRC and locality-aware comparisons as part of model triage now, especially once a copy/reuse mechanism exists.

### What I Would Do Next

1. Build a pretrain bank and treat pretrain selection as a first-class search problem.

2. Implement chunk-continuity or stitched-window training before trying more auxiliary losses.

3. Fork the roadmap by corpus instead of forcing one recipe:
   Tencent: higher-K regime or mixture prior, no low-K simplification.
   Alibaba: keep the validated low-K regime structure.

4. Spend the next bold architectural effort on explicit locality modeling:
   reuse/not-reuse decision, recent-object memory or copy path, stride generation only on non-reuse.

5. Use HRC plus reuse/locality metrics to rank candidate checkpoints, not just MMD/recall.

### Short Take

Reading the experiment history end to end, the repo has already done the hard part of eliminating a lot of seductive but low-value ideas. The winning story now looks much narrower: protect and search over strong pretrains, stop pretending a 12-step local objective will magically yield long-horizon dynamics, make multimodality explicit per corpus, and give locality an architecture that can actually copy rather than merely regress.

---

## Round 3

### Big Bets And Reopened Ideas

1. `[P1]` Reopen conditioning, but only as a redesigned conditioning space, not as another raw `cond_dim` bump. The negative result in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L110) says that `cond_dim=13` with the current feature set hurt badly, but the newer R pass now tells us why that test was dirty: the current vector still includes redundant or near-degenerate dimensions, especially `backward_seek_ratio` and `opcode_switch_ratio`, while the new informative candidates are `object_unique`, `signed_stride_lag1_autocorr`, and `obj_size_std` in [R-REBUTTAL.md](/Users/darrell/Zarathustra/R-REBUTTAL.md#L20) and [R-REBUTTAL.md](/Users/darrell/Zarathustra/R-REBUTTAL.md#L244). The bigger idea here is not "13 instead of 10"; it is an R-informed conditioning redesign:
   remove algebraically redundant and unlearnable dimensions,
   add only the high-signal ones,
   and compress them into an orthogonal factor space before they ever reach the generator.
   [R-REBUTTAL-RESPONSE.md](/Users/darrell/Zarathustra/R-REBUTTAL-RESPONSE.md#L74) explicitly notes that groupwise factor reduction is still missing. I think this deserves a clean retry.

2. `[P1]` Add a window-level analysis bridge and use it to supervise the regime model. The R stack now admits that it is still file-level rather than window-level in [R-ANALYSIS.md](/Users/darrell/Zarathustra/R-ANALYSIS.md#L914), and [R-REBUTTAL-RESPONSE.md](/Users/darrell/Zarathustra/R-REBUTTAL-RESPONSE.md#L240) says the same gap remains between family findings and actual training behavior. That creates a big opportunity: derive window-level pseudo-labels for burst regime, locality class, object-diversity band, and stride-memory pattern, then train the stage-1 regime selector or router against those labels. This is stronger than hoping the GAN discovers the structure unsupervised from 12-step windows alone.

3. `[P1]` Build an explicit locality engine, not just locality losses. The repo has already encoded reuse and stride separately, but the generator still does not have a mechanism for reuse, only a target for reuse. Given the persistent reuse gap in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L1314), the strongest new idea I would add is a pointer/copy style path:
   first predict `reuse / forward seek / backward seek`,
   then for reuse copy from a recent-object memory,
   and only regress stride magnitude when the model has committed to a non-reuse move.
   This is the architectural version of what [TODO.md](/Users/darrell/Zarathustra/TODO.md#L154) already argues philosophically: locality should be first-class. If this works, it is the clearest route to a real HRC jump.

4. `[P1]` Treat poison points as a modeling problem, not just a preprocessing problem. The clip defense in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L37) was the right emergency fix, and the R rebuttal makes clear that the stride outliers are operationally meaningful in [R-REBUTTAL.md](/Users/darrell/Zarathustra/R-REBUTTAL.md#L178). The next big step is to stop mixing the main mass and the poison tail in one undifferentiated training distribution. I would seriously consider a two-population strategy:
   train a main generator on the core distribution with outliers quarantined,
   then either fine-tune a specialized tail expert or sample from a separate outlier regime.
   This could rescue ideas that look unstable today simply because the training game is being periodically hijacked by a tiny number of toxic files.

5. `[P1]` Add a routed mixture-of-experts generator. This is not written explicitly in the repo, but it feels like the natural synthesis of the evidence already on disk. The experiments show that corpora differ, K differs by corpus, and some regimes are intrinsically hard. Instead of one generator plus one regime embedding table, use a small router driven by the cleaned conditioning factors or window pseudo-labels, and let different experts own different workload families:
   one expert for high-reuse/locality-heavy windows,
   one for burst-heavy random traffic,
   one for low-diversity read-dominated windows,
   and optionally corpus-specific experts for Tencent vs Alibaba.
   Conceptually, this is a stronger and more explicit version of the regime sampler in [IDEAS.md](/Users/darrell/Zarathustra/IDEAS.md#L79), and I think it fits the race better than continuing to push a single monolith.

6. `[P2]` Some previously abandoned ideas deserve a limited reopening because they were not cleanly tested:
   GMM prior: it was silently disabled in v41-v43, then confounded with `var_cond` in v44, and only later got a cleaner read in v45/v46, where it helped some but not enough in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L933) and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L539). I would not bet the race on it, but I also would not call it dead.
   Expanded conditioning: the failed `cond_dim=13` run happened before conditioning cleanup and orthogonalization, and even the version log says it may need progressive feature introduction or prenormalization in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L117). That is not a real falsification of the R-added features.
   Projection critic: still not a front-runner, but if it ever comes back it should be only in a late-stage fine-tune where the generator already has symmetric access to the same regime information and the critic is deliberately weakened.

7. `[P2]` Several other abandoned ideas still look genuinely low-priority, even after accounting for poison points:
   BayesGAN improved recall but repeatedly lost the MMD side of the game in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L394) and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L242).
   Mixed-type recovery still looks like the wrong hill to die on in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L382).
   Deeper stacked LSTMs failed after the clip and auto-drop era as well, so I would not reopen them unless they are embedded inside a more structured design rather than used as a raw depth increase.

### Additions To The Race Plan

1. Run one clean conditioning redesign experiment:
   drop `backward_seek_ratio` and `opcode_switch_ratio`,
   add the 3 R-identified features,
   and project the resulting vector into a low-rank factor space before use.

2. Build a window-level characterization pass for Tencent and Alibaba and use it to create regime pseudo-labels.

3. Prototype a reuse-aware generator path with recent-object memory or copy semantics.

4. Split training into core-distribution and poison-tail phases instead of forcing one model to absorb both from the start.

5. If there is appetite for one moonshot beyond that, make it a routed MoE regime-to-event generator, not another critic or loss variant.

### Short Take

The newer R work changes the interpretation of a few old experiments. Some things really were bad ideas, but some "failures" were actually tests run with redundant columns, poisoned tails, or the wrong abstraction level. The biggest opportunity now is to use the new analysis not just to tune the existing model, but to redesign what the model is allowed to condition on, what it is asked to copy, and which parts of the distribution it should learn jointly versus separately.

---

## Round 4

### Since-Review Check

1. `[P1]` No new commits have landed after this peer review. `HEAD` and `origin/main` are both still at the peer-review commit, so there is no later code change to assess yet. That matters because the strategic conclusions in Rounds 2 and 3 have not been superseded by a new model result.

2. `[P2]` There was, however, a meaningful evaluation-fix wave immediately before this review, and it partially addressed earlier concerns:
   HRC cache-fidelity was added in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py).
   eval-time generator construction now threads through `gmm_components`, `var_cond`, `n_regimes`, and `num_lstm_layers` in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L399).
   conditional eval now prefers char-file conditioning pools over noisy window descriptors in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L420).
   reuse-rate evaluation was partially fixed in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L539), in the sense that it no longer reads the obviously wrong feature.

3. `[P2]` Some earlier evaluation findings remain open even after that cleanup:
   the PRDC fallback is still wrong in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L314), and this still matters locally because `prdc` is not installed;
   the reuse-rate path still hardcodes column 3 rather than using `prep.col_names` in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L539);
   the baseline path still calls `_sample_fake()` without passing real windows in [llgan/eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L582), so conditional baseline comparisons are still fragile.

4. `[P2]` The actual project rebuttal is still unwritten. [REBUTTAL.md](/Users/darrell/Zarathustra/REBUTTAL.md#L1) remains a blank placeholder. That is separate from the R-side critique/response pair and should not be confused with it.

5. `[P2]` The R rebuttal response is real and useful, but it is analysis-side follow-through, not the main rebuttal for the project. [R-REBUTTAL-RESPONSE.md](/Users/darrell/Zarathustra/R-REBUTTAL-RESPONSE.md#L12) documents substantial progress:
   silhouette-based K diagnostics,
   per-regime feature attribution,
   outlier decomposition,
   top-N sensitivity summaries,
   block-vs-random diagnostics,
   and conditioning audits.
   That strengthens the evidence base behind the strategic recommendations in Rounds 2 and 3, but it does not change model behavior by itself.

6. `[P2]` The most important unchanged conclusion from the rebuttal side is still the file-level versus window-level mismatch. [R-REBUTTAL-RESPONSE.md](/Users/darrell/Zarathustra/R-REBUTTAL-RESPONSE.md#L9) says the biggest remaining gap is that the current R pass is family-level and file-level, while the GAN ultimately lives at the window level. That reinforces, rather than weakens, the recommendation to build a window-level characterization bridge and use it to supervise regime structure.

### Updated Read

The current state is not "the review is stale." It is closer to "the repo has improved its measurement surface, but has not yet changed the underlying strategic picture." The rebuttal-side analysis has become stronger, the evaluation harness is better than it was, and the main race question is still the same one: how to turn those newer insights into a model that handles long-horizon dynamics and locality, not just aggregate fidelity.

---

## Round 5

Since Round 4, the new commits have mostly updated [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md), [REBUTTAL.md](/Users/darrell/Zarathustra/REBUTTAL.md), [TODO.md](/Users/darrell/Zarathustra/TODO.md), and [IDEAS.md](/Users/darrell/Zarathustra/IDEAS.md), not the core Python paths. That matters because the biggest new information is not a new architecture. It is a reset in what the current code should make us believe.

### Reproducibility Reset

1. `[P1]` The model being trained is not the same conditional model being evaluated or emitted. During training, `_make_z_global()` applies the full conditioning stack: `cond_encoder`, `regime_sampler`, and conditioning-aware GMM noise in [train.py](/Users/darrell/Zarathustra/llgan/train.py#L83). But training-time checkpoint selection in [mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py#L254), offline eval in [eval.py](/Users/darrell/Zarathustra/llgan/eval.py#L438), and inference helpers in [model.py](/Users/darrell/Zarathustra/llgan/model.py#L631) bypass that path and mostly do a raw `torch.cat([cond, noise])`. [generate.py](/Users/darrell/Zarathustra/llgan/generate.py#L52) is even looser: it reconstructs `Generator` without threading through `film_cond`, `var_cond`, `n_regimes`, `gmm_components`, or `num_lstm_layers`, then also builds `z_global` by direct concatenation in [generate.py](/Users/darrell/Zarathustra/llgan/generate.py#L110). For the recent best recipes, which rely heavily on `n_regimes`, `var_cond`, and `gmm_components`, this means the repo is often ranking and exporting checkpoints under a different inference contract than the one used in joint training. I now think this is the single biggest technical risk in the project, and a plausible contributor to both the giant train/eval gaps and the failed verbatim controls in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L7).

2. `[P1]` The reproducibility controls are still confounded by random preprocessing and mild validation leakage. In multi-file mode, the preprocessor is fit on a random seed subset before the validation split is carved out in [train.py](/Users/darrell/Zarathustra/llgan/train.py#L368) and [train.py](/Users/darrell/Zarathustra/llgan/train.py#L395). Once clipping became the poison-point defense and auto-drop became part of the effective schema, that seed subset stopped being a harmless convenience and became part of the experiment definition. It changes min/max ranges, clipping boundaries, and even which columns exist. Because the fit happens before the held-out files are chosen, it can also see future validation files. So when [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L28) says `alibaba_v42` failed to reproduce `v37`, and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L7) says `tencent_v75` failed to reproduce `v68`, those are not clean seed-only controls. A fixed per-corpus preprocessor manifest should now be treated as part of the benchmark, not just infrastructure.

3. `[P1]` The repo is still selecting `best.pt` on an evaluation surface the session no longer trusts. The code still promotes checkpoints from EMA metrics on a 10-file validation tensor and `mmd_samples=1000` in [train.py](/Users/darrell/Zarathustra/llgan/train.py#L1583), after building that tensor from only 10 held-out files in [train.py](/Users/darrell/Zarathustra/llgan/train.py#L403). But the newest controls say that this surface is not reliably rank-preserving: both [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L7) and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L28) show training-log or EMA scores in the "record" range mapping to full-eval results 30-50% worse. Until checkpoint selection includes either a periodic shadow full eval on a larger fixed file bundle or a surrogate proven to match it, the project will keep promoting false winners and falsely killing ideas. At this point that is not a paper-cleanup issue. It is a race issue.

4. `[P2]` The new controls materially reopen some ideas, but not all of them. This is the substantive mistake from a few revisions back: several ideas were judged against an unreproducible benchmark and therefore closed too aggressively. Once the operational floors move to `0.142` on Alibaba and `0.1225` on Tencent in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L24) and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L40), [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L138) alibaba `block_sample` at `0.094` stops looking like a failure and starts looking like one of the strongest reproducible improvements in the repo, especially with its HRC gain. [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L126) Tencent `multi-scale critic` and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L86) Tencent `2-step supervisor` also deserve fair retests because they were killed or judged from training-log trajectories rather than full eval under the recalibrated floor. By contrast, feat-critic still looks truly closed across corpora because it fails even after the baseline reset and keeps producing the worst train/eval pathology in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L48) and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L67).

5. `[P2]` The session should stop treating one lucky single-seed run as enough evidence that an idea is either "the baseline" or "dead." The top of [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L7) now reads like a warning against point estimates. From here, I would rather spend compute on `3 seeds × 2 candidate families` than on `6 singleton knobs`. The unit of evidence should become a tiny seed bundle with the same preprocessor, the same held-out full-eval bundle, and the same checkpoint-selection policy. That is not slower in practice if it prevents another cycle of optimizing toward an unreproducible ATB.

6. `[P2]` A targeted Fourier analysis is worth doing even if it ends up being more diagnostic than directly predictive. [IDEAS.md](/Users/darrell/Zarathustra/IDEAS.md#L9) was right to leave the question open. I would specifically run spectral analysis on `ts_delta`, `obj_id_stride`, `obj_size`, and binary locality channels such as reuse, both on real traces and on synthetic long-rollouts. That can reveal burst periodicities, narrowband artifacts, and regime-specific cadence that MMD, PRDC, ACF, and even DMD-GEN can all miss or smear together. In a race, that matters because it may tell you whether a supposedly "bad" model is actually getting the right rhythms but the wrong marginals, or vice versa.

7. `[P2]` The highest-upside modeling story is still the one from Round 3, but the order of operations has changed. First stabilize measurement and conditioning semantics. Then push the two bets that still have the clearest path to a real gap: an explicit locality or copy mechanism, because reuse is still not being modeled structurally; and a window-level regime bridge, because the R work is still file-level while the GAN learns on 12-step windows. The bold ideas are still right. The repo now just also needs a benchmark contract strong enough to tell whether they worked.

### What I Would Do Next

1. Build one canonical `z_global` inference helper and route [train.py](/Users/darrell/Zarathustra/llgan/train.py), [mmd.py](/Users/darrell/Zarathustra/llgan/mmd.py), [eval.py](/Users/darrell/Zarathustra/llgan/eval.py), [model.py](/Users/darrell/Zarathustra/llgan/model.py), and [generate.py](/Users/darrell/Zarathustra/llgan/generate.py) through it before trusting any more conditional-model comparisons.

2. Freeze one per-corpus preprocessor manifest and one held-out full-eval bundle, then rerun the best reproducibility controls.

3. Reopen only three families under the new benchmark: Alibaba `block_sample`, Tencent `multi-scale critic`, and Tencent `2-step supervisor`.

4. Add a short Fourier-analysis pass to the evaluation workflow for `ts_delta`, `obj_id_stride`, `obj_size`, and reuse-related series, using both real traces and long synthetic rollouts.

5. After that, spend architecture effort on a reuse or copy path and window-level regime supervision, not another scalar sweep.

### Short Take

The last few days did change my view. The biggest story is no longer just "find a better model." It is that the repo is still partly training one conditional system, evaluating another, and benchmarking both on a moving preprocessing target. Fix that contract first, and several supposedly failed ideas become worth another shot. Ignore it, and the team risks winning training logs while losing the actual race.

---

## Round 6

### What I Think Of The New Numbers

1. `[P1]` The new records are real progress, and they validate the Round 5 warning that some ideas had been closed too aggressively after a benchmark mistake. [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L7) Tencent `v78` at `0.1008` and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L28) Alibaba `v46` at `0.1283` are the strongest evidence yet that revisiting old ideas under the recalibrated floor was the right call. This is not just noise. It is the first concrete payoff from that reopening logic.

2. `[P1]` The new numbers strongly reinforce the corpus-specific playbook. Tencent’s breakthrough came from `block_sample` layered onto the new `v76` base in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L13). Alibaba’s breakthrough came from raising regime capacity from 2 to 4 in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L34). Then [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L45) shows that 8 regimes is already too many for Alibaba. That is exactly the kind of asymmetric answer a race team wants early: stop looking for one universal recipe.

3. `[P1]` The new numbers also give a cleaner answer on one reopened idea: `supervisor-steps=2` now looks closed on both corpora. [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L55) Tencent `v77` fails badly at full eval, and [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L64) Alibaba `v45` never becomes competitive. I would now take this out of the active search space unless it is embedded inside a very different architecture.

4. `[P2]` Tencent `v78` is a major distributional win, but not yet a true dynamics/locality win. The record is driven by MMD² and recall holding up at eval in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L7), and that is important. But DMD-GEN is still `0.7416`, AutoCorr is still elevated, and HRC-MAE is `0.0795`, which is not a cache-fidelity breakthrough. My read is that `block_sample` fixed a training/eval mismatch and improved support coverage, but it did not solve the long-horizon law of the process. That makes the Fourier-analysis recommendation more valuable, not less, because the model may now be getting the right sample cloud while still missing the right rhythms.

5. `[P2]` Alibaba `v46` is the strongest evidence yet that regime count is a structural bottleneck rather than a cosmetic hyperparameter. Four regimes beat the reproducibility floor; eight regimes over-partitioned the corpus and hurt recall in [VERSIONS.md](/Users/darrell/Zarathustra/VERSIONS.md#L45). That is a meaningful architectural clue. It suggests the next Alibaba-side gains are more likely to come from a smarter regime or routing design than from another generic loss change. At the same time, `v46` still leaves DMD-GEN around `0.71` and Context-FID at `0.30`, so it should be read as a better workload-family fit, not as a full dynamics solution.

6. `[P2]` The new records do not rebut the main technical caveat from Round 5, because there were still no core code changes in the commits that produced this update. The repo got better numbers through better experiment selection, not through a resolved train/eval/generate contract. That means the wins are strategically meaningful, but the conditioning-path mismatch and benchmarking concerns still stand. In other words: the search has improved faster than the infrastructure.

### Updated Race Plan

1. Tencent should now have a clear primary branch: keep `block_sample` on, keep the `v76/v78` family as the base, and search on top of that.

2. Alibaba should now have a different primary branch: keep `n-regimes=4` as the working baseline and stop treating 2 and 8 as equally plausible defaults.

3. `supervisor-steps=2` should leave the active queue.

4. The sidecar metrics need to stay visible even when combined score improves:
   Tencent just showed that you can set a record while DMD-GEN and HRC remain unresolved.

5. A short Fourier-analysis pass is even more justified now, because the new best Tencent model improved distributional holdout behavior without improving the temporal story.

### Short Take

The new numbers are good news, and they are genuinely informative. They say the reopening was worth doing, `block_sample` is the current Tencent lever, and `n-regimes=4` is the current Alibaba lever. They also say the race is not over: the repo is getting better at matching distributions than at matching dynamics, and the infrastructure caveats from Round 5 still apply.
