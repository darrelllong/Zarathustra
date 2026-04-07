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
