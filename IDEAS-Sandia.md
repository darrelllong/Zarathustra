1	### 22. Hierarchical Trace Generation
2	
3	**Why this is the breakthrough candidate:** The current generator only plans a 12‑step window at a time, treating each window independently apart from hidden‑state carry. By introducing a coarse global plan that specifies high‑level workload characteristics (e.g., overall burstiness, average stride, total size budget) and letting a fine‑grained generator produce each window conditioned on that plan, we explicitly separate *what* the trace should look like from *when* events occur.
4	
5	**Implementation sketch:**
6	1. Add a lightweight global planner module (e.g., a small MLP that predicts a sequence of latent global vectors, one per window).  
7	2. Train the planner jointly with the generator using a multi‑head loss: the generator learns to match per‑window real data while the planner learns to produce a sequence that reconstructs the full trace using a *trace‑reconstruction* loss (e.g., MAE on aggregated statistics). 
8	3. During inference, sample a planner sequence from the prior and feed each latent into the window generator.
9	
10	**Expected impact:**  • Improves global coherence and reduces drift across window boundaries.  • Lowers reliance on hidden‑state carry, allowing the generator to be resetted each window.  • Makes training more stable and gives a clear signal that “global structure” matters.
11	
12	---
13	
14	### 23. Explicit Long‑Term Object Memory
15	
16	**Why this is the breakthrough candidate:** Reuse decisions are currently learned implicitly by the discriminator in the generator, which struggles to preserve long‑range object identity across hundreds of events. An explicit, trainable memory of recently accessed objects can give the generator a concrete way to decide *which* object to reuse.
17	
18	**Implementation sketch:**
19	1. Maintain a fixed‑size hash‑based memory table that stores a small embedding of each object seen in the last N events (e.g., N=256).  
20	2. The generator outputs a *reuse gate* probability and, if reusing, a query vector.  
21	3. During training, use a differentiable attention mechanism (softmax over memory similarity) to select an object.  
22	4. Add a reconstruction loss that encourages the selected object's features (size, opcode) to match the generated values.
23	
24	**Expected impact:**  • Significantly raises reuse rate and fidelity of reuse patterns.  • Reduces cache miss variance, making the generated traces more realistic for downstream work‑loads.  • Provides an interpretable interface for inspecting which objects are being reused.
25	
26	---
27	
28	### 24. Cache‑Aware Training Loss
29	
30	**Why this is the breakthrough candidate:** The evaluation uses an LRU cache simulator to compute HRC‑MAE, but the generator does not see cache behavior during training. Incorporating a differentiable cache simulation (or a learned cache‑behavior predictor) into the loss encourages the generator to produce traces that are cache‑accurate.
31	
32	**Implementation sketch:**
33	1. Implement a lightweight, differentiable LRU (or splay) cache simulation built from tensor ops.  
34	2. During training, run a short‑run of the simulator on a batch of generated windows and compute the HRC‑MAE against the target HRC curve derived from the char‑file.  
35	3. Backpropagate this loss together with the standard GAN losses.
36	
37	**Expected impact:**  • Directly aligns training with the downstream metric.  • Dramatically improves cache hit‑ratio fidelity, which is the main gap for full‑trace realism.  • Helps the GAN avoid generating long sequences that locally look correct but globally produce poor cache behavior.
38	
39	---
40	
41	### 25. Window‑Boundary Alignment Loss
42	
43	**Why this is the breakthrough candidate:** Even with state carry, the model can drift when concatenating windows, producing subtle inconsistencies at boundaries. A simple alignment loss that penalizes discontinuities in key statistics (stride, size, reuse rate) between adjacent windows can guide the generator to produce smoother transitions.
44	
45	**Implementation sketch:**
46	1. After generating two adjacent windows, compute their aggregate statistics (mean stride, total size, reuse rate).  
47	2. Compute an L1 loss between each statistic of window t and the corresponding statistic of window t+1.  
48	3. Add this loss term to the overall generator objective with a small weight.
49	
50	**Expected impact:**  • Produces more natural long‑trace flows.  • Reduces the need for manual chunk stitching and makes the generator's output directly usable for long‑trace evaluation.
51	
52	---
53	
54	## Bug 2 (open) — cuDNN RNN backward through eval‑mode network
55	Diagnosed in PEER‑REVIEW‑Sandia.md Round 38. No fix prescribed. Need investigation before training can proceed past Phase 4. Added to IDEAS‑Sandia.md for future work.
