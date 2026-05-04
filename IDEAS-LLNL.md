### 22. Hierarchical Trace Generation

**Why this is the breakthrough candidate:** The current generator only plans a 12‑step window at a time, treating each window independently apart from hidden‑state carry. By introducing a coarse global plan that specifies high‑level workload characteristics (e.g., overall burstiness, average stride, total size budget) and letting a fine‑grained generator produce each window conditioned on that plan, we explicitly separate *what* the trace should look like from *when* events occur.

**Implementation sketch:**
1. Add a lightweight global planner module (e.g., a small MLP that predicts a sequence of latent global vectors, one per window).  
2. Train the planner jointly with the generator using a multi‑head loss: the generator learns to match per‑window real data while the planner learns to produce a sequence that reconstructs the full trace using a *trace‑reconstruction* loss (e.g., MAE on aggregated statistics). 
3. During inference, sample a planner sequence from the prior and feed each latent into the window generator.

**Expected impact:**  • Improves global coherence and reduces drift across window boundaries.  • Lowers reliance on hidden‑state carry, allowing the generator to be resetted each window.  • Makes training more stable and gives a clear signal that “global structure” matters.

---

### 23. Explicit Long‑Term Object Memory

**Why this is the breakthrough candidate:** Reuse decisions are currently learned implicitly by the discriminator in the generator, which struggles to preserve long‑range object identity across hundreds of events. An explicit, trainable memory of recently accessed objects can give the generator a concrete way to decide *which* object to reuse.

**Implementation sketch:**
1. Maintain a fixed‑size hash‐based memory table that stores a small embedding of each object seen in the last N events (e.g., N=256).  
2. The generator outputs a *reuse gate* probability and, if reusing, a query vector.  
3. During training, use a differentiable attention mechanism (softmax over memory similarity) to select an object.  
4. Add a reconstruction loss that encourages the selected object's features (size, opcode) to match the generated values.

**Expected impact:**  • Significantly raises reuse rate and fidelity of reuse patterns.  • Reduces cache miss variance, making the generated traces more realistic for downstream work‑loads.  • Provides an interpretable interface for inspecting which objects are being reused.

---

### 24. Cache‑Aware Training Loss

**Why this is the breakthrough candidate:** The evaluation uses an LRU cache simulator to compute HRC‑MAE, but the generator does not see cache behavior during training. Incorporating a differentiable cache simulation (or a learned cache‑behavior predictor) into the loss encourages the generator to produce traces that are cache‑accurate.

**Implementation sketch:**
1. Implement a lightweight, differentiable LRU (or splay) cache simulation built from tensor ops.  
2. During training, run a short‑run of the simulator on a batch of generated windows and compute the HRC‑MAE against the target HRC curve derived from the char‑file.  
3. Backpropagate this loss together with the standard GAN losses.

**Expected impact:**  • Directly aligns training with the downstream metric.  • Dramatically improves cache hit‑ratio fidelity, which is the main gap for full‑trace realism.  • Helps the GAN avoid generating long sequences that locally look correct but globally produce poor cache behavior.

---

### 25. Window‑Boundary Alignment Loss

**Why this is the breakthrough candidate:** Even with state carry, the model can drift when concatenating windows, producing subtle inconsistencies at boundaries. A simple alignment loss that penalizes discontinuities in key statistics (stride, size, reuse rate) between adjacent windows can guide the generator to produce smoother transitions.

**Implementation sketch:**
1. After generating two adjacent windows, compute their aggregate statistics (mean stride, total size, reuse rate).  
2. Compute an L1 loss between each statistic of window t and the corresponding statistic of window t+1.  
3. Add this loss term to the overall generator objective with a small weight.

**Expected impact:**  • Produces more natural long‑trace flows.  • Reduces the need for manual chunk stitching and makes the generator's output directly usable for long‑trace evaluation.

---

### 26. Atlas-Fit IRD-Shape Loss (R284 followup)

**Why this is the breakthrough candidate:** R284.X (R270 capacity, 192 states) and R284.Y (low capacity, 6 states) both showed the LLNL atlas hits a per-trace floor of ~0.10 LRU HRC-MAE on alibabaBlock_521 — *capacity is not the bottleneck.* 2DIO achieves 0.02–0.05 by directly parameterizing the IRD distribution with a 3-param analytical model. The LLNL atlas approximates that shape indirectly via state-conditioned rank PMFs and pays a 2–5× MAE penalty for the indirection. The fix is to add a direct objective on the IRD shape during atlas fit.

**Implementation sketch:**
1. During `neural_atlas fit`, after each epoch, generate a small batch (e.g. 100k records) from the current atlas using the manifest's actual eval cache sizes.
2. Compute LRU stack-distance histogram on both real and generated batches; sum-MAE over the 43 fine bins (or directly over the cumulative HRC at the eval sizes).
3. Add this MAE as an auxiliary loss term, weighted ~0.1× the trans_loss, with a slow ramp-up after epoch 100 (so the rank PMF is roughly fit before the IRD-shape gradient kicks in).
4. *Alternative (cheaper):* Maintain a running bootstrap estimate of the real IRD shape and project the generator's predicted PMF onto that target via KL/EMD instead of running cachesim per epoch.

**Expected impact:** • Closes some-or-all of the 2DIO per-trace gap without abandoning the corpus-fit strength. • Gives an architectural path to *learn* what 2DIO hand-designs — atlases that are simultaneously corpus-general and IRD-faithful. • Worst case: confirms the gap is fundamental (which is itself a paper).

**Risk:** generating per-epoch eval batches is expensive; mitigations are to keep the inner batch small (100k), eval only every K epochs, and gate behind a `--atlas-fit-hrc-loss` flag so existing recipes stay unchanged.

---

