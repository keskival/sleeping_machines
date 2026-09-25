# Related work — where our ideas already exist

Written 2026-09-25 from a quick targeted search, not a systematic review. "Not found"
means not found in these searches. Every entry needs reading in full before we claim
anything about novelty.

| Our idea (THEORY.md) | Existing work | Overlap |
|---|---|---|
| computing with arrival times; first arrival wins | Race logic (Madhavan, Sherwood & Strukov, ISCA 2014); space-time algebra / temporal neural networks (J. E. Smith, ISCA 2018) | very close for the forward mechanism; cite as its origin |
| exact gradients along the realised event path (§3) | EventProp (Wunderlich & Pehle 2021); SpikeProp (Bohte et al. 2002); TTFS backprop (Mostafa 2017; Comsa et al.; Göltz et al. 2021) | the same; we only restate it |
| the gradient misses spike creation and deletion (§4, §14.1) | known limitation of EventProp; loss shaping (Nowotny et al., Neuromorph. Comput. Eng. 2025); smooth exact gradients with spikes that only appear or vanish at trial end (Klos & Memmesheimer, PRL 2025) | the problem is recognised; the remedies differ from ours |
| credit for near-threshold neurons as causal ("but-for") effect (Δ, §13) | spike discontinuity estimation (Lansdell & Kording, PLOS Comput Biol 2023) | closest conceptual match; theirs is a statistical estimate over trials with reward, ours a deterministic residue of cancellation in one trial |
| surrogate gradient = expected spiking under escape noise (§4) | Gygax & Zenke, Neural Computation 2025 | the same claim |
| "fires by time c" is linear in the weights; projection learning (§11.1, M13) | Tempotron (Gütig & Sompolinsky 2006) rests on linearity in the weights at fixed times; Linear Constraints Learning for Spiking Neurons (Nguyen & Chu 2021) | largely known; our cancellation-aware PA variant is a small step |
| minimal single-unit repair (§13, M18) | Madaline Rule II (Widrow et al. 1988), minimal disturbance | close in spirit; new: time, half-space costs, delegation to causes and cancellers |
| gradients through event order; path-local AD gradients (§14) | Differentiable discrete event simulation for queueing networks (Che et al. 2024) | close, from the simulation side; warns that naive softmin smoothing drifts |
| speculative events (§10, §14.6) | Time Warp / speculative distributed simulation of SNNs (SIGSIM 2022) | speculation for parallel simulation, not for learning |
| random-feedback hidden credit that works without following the gradient (FINDINGS: alignment puzzle) | feedback alignment (Lillicrap et al. 2016); direct feedback alignment (Nøkland 2016); direct random target projection, DRTP (Frenkel et al. 2021) | our crl_fa is DFA-like; the "template mechanism" hypothesis is essentially DRTP's; tested as variant crl_drtp |
| targets passed down to hidden units | target-based spiking learning (e.g. arXiv 2002.05619); target propagation | related |

## Not found in these searches

- A learning rule defined as a choice of semiring (sum-product, min-sum, single leaf) over
  the network's own tree of event histories.
- Shadow-event beams for learning, with collapse residues waiting for their branch's result.
- Cancellation residues as the sufficient statistic for boundary terms.
- Recruitment triggered when no residue is within reach of fixing the error.
- Work (energy) as a trainable objective through cancellation times (spike-count
  penalties exist, but are a different thing).

## Consequences

- The forward mechanism and exact event gradients are established. Our contribution
  must be on the learning side, built around cancellation.
- Position against Lansdell & Kording and against the EventProp creation/deletion work;
  use loss shaping and Klos & Memmesheimer as baselines in M19.
- Present §11.1's half-space property as a known fact we use, not a discovery.

## Sources

- https://www.nature.com/articles/s41598-021-91786-z (EventProp)
- https://iopscience.iop.org/article/10.1088/2634-4386/ada852 (loss shaping)
- https://arxiv.org/abs/2309.14523 (Klos & Memmesheimer)
- https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011005 (Lansdell & Kording)
- https://arxiv.org/pdf/2404.14964 (Gygax & Zenke)
- https://arxiv.org/abs/2103.12564 (Nguyen & Chu)
- https://www-isl.stanford.edu/~widrow/papers/c1988madalinerule.pdf (MRII)
- https://arxiv.org/abs/2409.03740 (Che et al.)
- https://dl.acm.org/doi/10.1145/3518997.3531027 (speculative SNN simulation)
- https://dl.acm.org/doi/abs/10.1109/ISCA.2018.00033 (space-time algebra)
- https://www.arch.cs.ucsb.edu/neuromorphic (race logic)
- https://www.nature.com/articles/s42256-021-00388-x (Göltz et al.)
- https://arxiv.org/pdf/2002.05619 (target spiking patterns)
