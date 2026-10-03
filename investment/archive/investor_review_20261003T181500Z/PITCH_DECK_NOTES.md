# Sleeping Machines — full pitch deck and diligence notes


Proposed raise: €3M. Bullish negotiating case: €50M priced pre-money; €100M stretch scenario.


This supersedes the pricing proposal in the older $10M discussion memo; it does not add new benchmark evidence.


All financial outcomes, budgets and milestone timelines are assumptions. No customer interest has been reported. Repository is private by founder instruction on 3 October 2026. This deck is a private review artifact; distribution and any future publication require a considered disclosure decision.


Research cut-off: completed records available on 3 October 2026. No pending training scores enter the deck.


## 1. Sleeping Machines

A trainable substrate for intelligence through time and sparse events.

Private review deck. The founder proposes a €3M raise. €50M priced pre-money is the central bullish negotiating thesis developed here; €100M is a stretch scenario, neither an independent fair-value appraisal nor an investor offer. The research evidence is exploratory and single-seed. No customer interest has been reported. No investor, vendor or customer has been contacted in preparing this deck.


## 2. A new model family. A new computing substrate.

The goal: useful intelligence with less active work, persistent state and adaptation.

Core mechanisms remain explicit: computational delays and temporal races; sparse addressed memory updates; small messages that mix input and persistent state; separate keys and values; credit to unrealized routes; silence-aware supervision where applicable. The newest language core implements a factorized temporal law, hard selection, deep local memories and route credit. It does not yet cover every general event-substrate mechanism. Future clockless hardware is a proposed implementation route, not a fabricated asset.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 3. The constraints reach from power grids to robot batteries.

Economic context for efficient intelligence; these figures are not our revenue market.

IEA projects all datacenter electricity, not only AI demand; GSMA subscribers are not annual device shipments; IFR reports industrial-robot installed stock, not all robotics. These denominators cannot be summed into a TAM. The thesis is an inference about potential demand for verified efficient computation. Incumbents are also improving rapidly, which raises the threshold for a commercially useful alternative.

- [M1: IEA — Key Questions on Energy and AI, executive summary](https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary) — 485 TWh datacenter electricity in 2025; central projection 950 TWh in 2030. Sector context, not addressable revenue.

- [M2: GSMA — The Mobile Economy 2026](https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-economy/wp-content/uploads/2026/02/The-Mobile-Economy-2026.pdf) — 5.8 billion unique mobile subscribers and 8.8 billion connections. Subscribers are not annual device sales or customers.

- [M3: IFR — World Robotics 2026 release, 24 September 2026](https://ifr.org/ifr-press-releases/news/five-million-robots-now-operate-in-factories-globally) — 5 million operating industrial robots in 2025 and more than 600,000 annual installations; no adoption by this project.


## 4. Hard sparse computation that can learn its alternatives.

Conceptual event path; the trained language core is implemented in a synchronous software emulator.

This construction combines familiar vector operations with temporal computation and sparse state. Separate keys determine addressing; values determine delivered content. Counterfactual credit gives unrealized alternatives a learning signal without softening the hard forward route. The present implementation is not a physical asynchronous machine: autograd, shared training windows, clipping and Adam remain. Winner-only inference still scores all candidate keys.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 5. Three results make the architecture worth scaling.

Completed text8 experiments; all numbers remain exploratory single-seed evidence.

T256 test values come from completed held-out evaluations; training scores and pending 90M results are excluded. One nominal pass uses about 9.994M training positions. A single seed and differing native/control training order limit generalization. The p96 improvement over LSTM is only 0.0081 bpc, with 2.89 times fitting work; it establishes a promising quality direction, not an iso-quality resource win.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 6. A promising quality–work frontier at small scale.

Same text8 held-out region and T256 windows; saved one-pass controls, not optimized frontier baselines.

p32 / D4 credited: 2.371491 bpc, 7.243 estimated whole-fit TFLOPs, 108,875 parameters. Saved Transformer-256×2: 2.426909, 111.262 TFLOPs, 1,658,907 parameters. This gives 15.36× lower fitting arithmetic and 15.24× fewer parameters for this saved control, with 0.0554 lower bpc. The strong one-pass LSTM is 2.170597 at 20.306 TFLOPs. Best native p96 is 2.162463 at 58.648 TFLOPs. A modern tuned Transformer/MoE/SSM and larger-data matched protocol are still needed.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 7. Learning the alternatives changes the result.

The same p32 / D4 model, data budget, seed and evaluation window.

The local linearized credit estimator adds a derivative pathway for alternative delivered values while preserving the forward winner. It is not the exact full counterfactual trajectory gradient. The paired p32 / D4 T256 gain is 0.134895 bpc, with 0.3034% extra whole-fit estimated operations. The corresponding p32 / D8 gain is 0.130367. Attribution is stronger than a comparison between different widths, but a single seed still leaves variance and hyperparameter sensitivity unresolved.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 8. More available state. The same eight selected writes.

Matched credited p32 / D4 pool expansion: 16 → 32 slots.

Pool 2 → 4 changes memory scalars from 512 to 1,024 and parameters from 108,875 to 177,019. At T256, quality changes from 2.371491 to 2.345157. Selected writes remain eight per input position; candidate keys increase 16 → 32. Winner-only per-position arithmetic increases 0.6576%, but whole-fit estimated work rises 64.9917%. All memory and key scoring still cost resources. A negative uncredited pool expansion was measured at D8, while this positive credited pair is D4; this is not a full matched 2×2 interaction test.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 9. The sparse path is implemented; system proof is next.

Traced arithmetic per input position. Only selected proposal values are computed in the winner path.

Small random float64 fixtures match full-emulator logits within roughly 1e-10. Actual trained float32 winner/state/cache parity and a full held-out rescore are prepared and pending. The prepared CPU worker retains prepacked matrices per fixed model version, but native admission remains unrun. Candidate key scans, copying, residency, setup, gather traffic, cache invalidation and quality must all be included in serving comparisons. The deck deliberately does not attach the trained quality to an unverified production backend or convert FLOPs to joules.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 10. An adaptation signal exists. Its work is accounted for.

An earlier integrated CPU model, on a new development stream; separate from the latest language fits.

Frozen versus online evaluation scores 8,191 targets with predict-before-update, 16-character delayed-feedback blocks, lr 1e-4, fresh Adam, clipping and full-backbone gradients. Both arms retain persistent local state. BPC improves by 0.095120; total processing-operation estimates increase from 210,902,697 to 2,264,163,359, a 10.7356× ratio. This supports trainability under adaptation, not inexpensive lifelong learning. Retention, drift, long-horizon memory and local asynchronous learning need new evidence. Loihi 2 already provides an on-chip learning precedent.

- [R2: Private online-learning experiment, 8,191 targets](pitch_deck_evidence_20261003.json) — Predict-before-update pilot: quality and complete processing-operation estimates; one checkpoint, stream and learning rate.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 11. The contribution is the complete construction.

Established alternatives already address parts of this problem; they remain essential controls.

The pitch does not assert sole invention of sparse activation, gating, normalization, key/value separation or hardware learning. It proposes an integrated trainable temporal substrate and evaluates consequences. The latest native language model has no dense carrier or n-gram count component, but remains a restricted software realization. A weak result in one restricted variant should be diagnosed; genuine superiority requires matched strong controls. Intel Loihi 2 is cited on the adaptation and strategic slides.

- [T1: Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Primary architecture reference. The deck’s small saved control is not a frontier-performance claim.

- [T2: Fedus et al. — Switch Transformers](https://arxiv.org/abs/2101.03961) — Capacity beyond selected computation has established MoE precedent; sparsity alone is not a novelty claim.

- [T3: Gu and Dao — Mamba](https://arxiv.org/abs/2312.00752) — Selective recurrent state-space sequence models are relevant competitive controls. No matched modern SSM result is claimed.


## 12. Enter through software. Earn the hardware expansion.

Proposed commercialization sequence; no customers or partner interest yet.

The first commercial workload is a selection hypothesis rather than a confirmed customer request. A staged software route lowers dependency on new silicon and permits independent benchmarking. A clockless implementation could benefit from less global synchronization and local state, but clock precision, signal fanout, memory, interconnect and fabrication overhead may dominate. Hardware economic claims require measurements. Costs for an ASIC tapeout are not estimated or funded here.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 13. Incumbents have reasons to evaluate a new substrate.

Potential counterparties only. Strategic relevance is an inference; no relationship is claimed.

AMD is the strongest identified public fit precedent, not a presumed buyer. Its Ventures page lists Series A–C focus, so a €3M research-stage round may need angels/specialist deep-tech leads before strategic institutional investment. Intel’s system is a research prototype; Google has substantial internal alternatives and could build rather than license. Possible routes are benchmark collaboration, co-development, licensing, investment or eventual acquisition; none is a current pipeline asset. No company logos or implied endorsements are used.

- [S1: AMD — Silo AI acquisition completion, 12 August 2024](https://www.amd.com/en/newsroom/press-releases/2024-8-12-amd-completes-acquisition-of-silo-ai-to-accelerate.html) — Announced approximately $665M all-cash acquisition; Poro and Viking on AMD. Mature-team strategic precedent, not a pre-seed valuation comparable.

- [S5: Intel — Hala Point research system, 17 April 2024](https://www.intel.com/content/www/us/en/newsroom/news/intel-builds-worlds-largest-neuromorphic-system.html) — Loihi 2 neuromorphic research prototype, event computing and prospective continuous learning. No Intel interest in this project claimed.

- [S6: Google — Eighth-generation TPU architecture announcement](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) — TPU 8t/8i, model–hardware co-design and sparse MoE infrastructure. Existing internal capability creates both potential fit and competition.


## 14. AMD’s Silo AI deal supports the strategic logic.

Announced approximately $665M all-cash acquisition in 2024; a mature-company precedent.

AMD’s announcement connects the acquisition to end-to-end AI solutions, engineering expertise, enterprise customers and Poro/Viking model work using AMD platforms, including LUMI. It does not establish that chip usage alone caused the acquisition. Silo had a large team and existing customers; neither applies here. The approximately $665M announced transaction value and Liquid AI’s $250M Series A amount are different quantities and are not converted into our pre-money valuation. They demonstrate strategic precedent, not an offer, floor or near-term acquisition expectation.

- [S1: AMD — Silo AI acquisition completion, 12 August 2024](https://www.amd.com/en/newsroom/press-releases/2024-8-12-amd-completes-acquisition-of-silo-ai-to-accelerate.html) — Announced approximately $665M all-cash acquisition; Poro and Viking on AMD. Mature-team strategic precedent, not a pre-seed valuation comparable.

- [S2: AMD — Acquisition announcement and LUMI collaboration, 10 July 2024](https://www.amd.com/en/newsroom/press-releases/2024-7-10-amd-to-acquire-silo-ai-to-expand-enterprise-ai-sol.html) — AMD highlighted enterprise solutions, expertise, software and Poro/Viking training on LUMI using AMD hardware; no single exclusive acquisition cause established.

- [S3: Liquid AI — $250M Series A announcement, 13 December 2024](https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai) — Financing announcement and AMD collaboration on efficient models; funding amount is not a disclosed company valuation.


## 15. Capture a fraction of verified value created.

Illustrative annual economics, contingent on quality and complete system savings.

Eligible spend is the subset of workload cost we can actually influence; it is not total hyperscaler capex or global electricity spend. Savings must be measured at equivalent quality, latency and service conditions. Capture share is a pricing assumption. €1B × 20% × 20% yields €40M annual license revenue, leaving €160M buyer benefit before transition overhead. 100M new devices × €0.50 is €50M device-cycle revenue, not subscriber-stock recurring ARR. A robotics illustration of one million paying systems at €50/year gives €50M/year, but assumes a large unsupported adoption level. These routes may overlap and should not be summed.


## 16. A founder at the intersection of models and systems.

Public professional record and a primary published invention document.

The XING timeline lists Cybercom December 2012–June 2018, HERE July 2018–April 2022, Alloy May 2022–August 2023, and Kaiko September 2023–December 2025. The founder’s portfolio lists Sleeping Machines and FAS Simulator. The primary EPO application EP4148389A2, published 15 March 2023, names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. It supports relevant invention experience, not a patent grant or an asset owned by this venture. Public personal biography and university links have availability/staleness issues; no degree or current-employment claim is inferred from them. Founder availability and references remain diligence items.

- [F1: Tero Keski-Valkama — public professional timeline](https://www.xing.com/profile/Tero_KeskiValkama) — Self-reported historical roles: Cybercom, HERE, Alloy.ai, Kaiko.ai. Dates and current commitment require founder confirmation.

- [F2: European Patent Office — published application EP4148389A2](https://patentimages.storage.googleapis.com/25/dd/41/af29b8e1391162/EP4148389A2.pdf) — Primary published document names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. Experience evidence, not Sleeping Machines-owned IP or proof of a grant.

- [F3: Tero Keski-Valkama — public project portfolio](https://keskival.github.io/) — Self-published portfolio lists Sleeping Machines and FAS Simulator; project access may have changed. Sole-founder status supplied by founder.


## 17. Build a moat around evidence, execution and integration.

The repository is now private; future progress disclosure needs a deliberate policy.

Tero instructed on 3 October 2026 that the GitHub repository was made private. This deck is a local private artifact and no public publication is authorized. Earlier public disclosure cannot be undone by repository privacy. README credits Tero Keski-Valkama and Karoliina Salminen as research authors; sole-founder status does not establish sole authorship, inventorship or IP ownership. Contributor agreements, intended licenses and employment-related rights must be resolved before presenting an exclusive technology asset. No granted patent portfolio, customer integration or defensible legal monopoly is claimed.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.

- [F2: European Patent Office — published application EP4148389A2](https://patentimages.storage.googleapis.com/25/dd/41/af29b8e1391162/EP4148389A2.pdf) — Primary published document names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. Experience evidence, not Sleeping Machines-owned IP or proof of a grant.


## 18. Use capital to retire the decisive uncertainties.

Milestones are proposed gates, not completed deliverables or guaranteed dates.

The timing is a planning assumption measured from funding. Three seeds and tuned modern controls are validation targets, not present assets. The ≥20% complete-cost gate is a proposed commercial screen and should be adapted to actual service requirements before experiments. Model quality and resource accounting must stay jointly comparable. Current owner queues retain priority; no new training or unguarded compute is launched to prepare this deck. Continued funding depends on completed results, measurable economics and a credible next workload rather than predictions entered as benchmark evidence.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 19. €3M to scale research and engineering aggressively.

Proposed 18-month budget; assumptions to refine with hiring and compute quotes.

The proposed allocation totals exactly €3M: team €1.35M, compute/replication €0.90M, hardware feasibility/measurement €0.25M, legal/IP/operations €0.20M, reserve €0.30M. Average six FTE × €150k/year fully loaded × 1.5 years equals €1.35M; this is a staffing assumption, not an actual hiring plan or salary survey. Operating burn excludes reserve; total budget consumption would average about €166.7k/month over 18 months if reserve is spent. Stage compute purchases by quality/system gates. Allocate founder plus representation/credit, systems/compiler, evaluation and hardware-measurement skills; exact staffing mix and ramp remain open.


## 20. Price the platform option with explicit assumptions.

€50M is a bullish negotiating case; €100M is a stretch case requiring substantially stronger conviction.

The illustrative model discounts the retained fraction of a successful future equity outcome to today and assigns zero failure value. It is not a complete corporate DCF, does not model all cash flows, and assumes the retained cohort benefits from an exit equity value after future financing. €10B exit × 30% retention / 1.15^10 is the conditional present value; €50M requires 6.7426% and €100M requires 13.4852%. These are neither forecasts nor inferred from small benchmarks. A €3M priced round at €50M pre-money gives 5.6604% initial investor ownership before option-pool changes, fees, preferences or future rounds; at €100M it gives 2.9126%. An investor must independently accept technology scalability, commercial value capture, rights and execution to support a premium price.


## 21. Successful platform adoption can support very large outcomes.

Conditional commercial arithmetic illustrates the scale required; no forecast or market multiple claim.

The €10B/€50B equity scenarios use deliberately stated revenue multiples rather than observed comparables. They require economics, durable margins and value capture that are wholly unproven. Multiplying revenue by a selected multiple is not an enterprise/equity reconciliation: the examples assume negligible net debt at exit. The scenarios overlap and are alternatives, not additive. Infrastructure relevance can justify funding risky research, but cannot itself prove a current valuation. Smaller niche success, delayed commercialization, licensing-only outcomes, further dilution and total failure remain possible.


## 22. Fund the evidence that makes a new substrate investable.

Private discussion draft · €50M pre-money bullish case · No customer interest reported.

Requested financing is €3M. The financing purpose is an aggressive but gated increase in experiment throughput and engineering capacity. The key next investor asset is replicated quality plus a measured service-level advantage; a clockless-chip thesis adds upside after mapping is credible. This private deck includes an appendix for diligence rather than presenting pending work as completed. Contact details, company identity, jurisdiction, cap table and legal terms should be supplied by the founder before distribution. No outbound messages or public release were made.


## 23. Quality and work, in the same units for every model.

Completed one-pass 10M text8 fits; T256 held-out evaluation; estimated operation counts.

All fitting columns include estimated complete step arithmetic rather than forward-only model FLOPs. Native representative operator traces include backward, alternative credit, clipping and Adam; dense controls use shape formulas with backward approximately twice forward. Inference includes traced unit and special-function arithmetic, but not DRAM traffic, copying, allocation, RNG, evaluation overhead or physical energy. Formula conventions differ and are stated. Per-position columns use one shared denominator type per column. Winner-only arithmetic is explicitly separated from emulator quality. Full CSV contains all 13 native fits and both controls.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 24. The comparison boundaries are part of the evidence.

Maintain the original records and disclose what a result actually measures.

The native score pays overlap-window warmup, evaluating roughly twice as many input positions as scored targets at T256. LSTM recurrent evaluation is a different protocol. Native T256 coverage is 999,936 targets, while original saved controls have a slightly different tail. The raw difference is small but must be disclosed. The 90M owner campaign is active; checkpoints count presented targets, including repeated random segments, and do not establish a final 90M held-out result. No pending cells are filled with expectations. No additional Transformer or LSTM training is launched here; the user reserved it for AWS.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 25. Keep the failures beside the positive mechanism results.

A disciplined research program should revise interpretations when evidence changes.

D8 uncredited pool 2 scores 2.4565 and pool 4 2.4981 at T256. The D4 credited pool pair is distinct and should not be sold as a fully matched interaction. Normalized read/write credit gives 2.3840 versus read-credit 2.3715; other write-credit variants diverge and are retained historically. Best p96 quality gains over LSTM with more work. DVS small screens suggested lower coarsening work, but the full all-seed quality gate failed and the strong RBF control remains better. Target-dependent E63/E79 mixtures are quarantined for leakage and never used in this deck.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report. Temporal races, counterfactual credit and native sparse inference; trained-weight sparse parity remains pending.


## 26. The price depends on dilution, exit scale and time.

Required platform-success odds at 10 years and 15% discount; zero failure value.

This table is generated from the frozen financial model, not entered by hand. At a €3B outcome and 10% retained equity, the required probabilities are much higher than at a €30B exit with 50% retention. A lower discount rate increases the present value, while delays and dilution reduce it. In the €3M/€50M round, an investor retains 1.6981% at exit if 30% of its initial stake survives, giving €169.81M or 56.60× gross MOIC in the assumed €10B success outcome. That conditional upside is not expected return; failure is zero and cash-flow timing/preferences are omitted.


## 27. A private evidence pack and primary external context.

Source labels in slide footers are clickable; companion files must travel with the PDF.

Raw research evidence is private, per founder instruction. Original completed result SHA256 hashes are in pitch_deck_evidence_20261003.json. The status PDF contains detailed theory and historical evidence; do not treat invalid-protocol archives or old rendered report scores as active claims. Market sources were accessed on 3 October 2026. These external sector denominators support relevance, not customer intent or a venture revenue forecast.


## 28. Public precedents establish relevance, not endorsement.

Historical roles are self-reported; inventorship is checked against the primary published document.

The founder profile and portfolio are linked from the founder slide (F1, F3). Transformer and Mamba references are linked from the competition slide. The private diligence checklist remains open: corporate entity, cap table, founder commitment, contribution chain including Karoliina Salminen, employer invention assignments, licenses, compute/hiring quotes, trained backend admission, replication and measured service advantage. A private repository does not rescind earlier disclosures or itself establish patentability. No permission to distribute or contact potential partners is inferred from preparing this deck.


## Reproduce and inspect

The editable slide narrative is PITCH_DECK.json. The frozen parent hashes, exact derived metrics and assumptions are in pitch_deck_evidence_20261003.json. CSV exports use identical column units for every model. Run the preparation only when deliberately updating the evidence cut-off; use a fresh publication tag to render. This build imports no numerical model runtime.


Open diligence: company/jurisdiction, cap table and option pool; founder availability; contribution and IP chain of title (including credited co-author Karoliina Salminen); employer invention assignments; intended software/model licenses; budget quotes; three-seed modern controls; trained sparse-backend parity; measured system energy/traffic; design-partner willingness to pay. No contacts have been made by this work.
