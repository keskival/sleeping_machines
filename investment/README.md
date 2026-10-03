# Private investment materials

Start with the **[16-slide investor pitch](sleeping_machines_pitch_deck_main.pdf)**.
The **[full 32-slide deck](sleeping_machines_pitch_deck.pdf)** adds 16 optional
technical and financial diligence slides. The current discussion draft proposes
a **€3M raise**, with **€50M priced pre-money as the bullish negotiating
case** and €100M as a separate appendix stretch scenario. Neither price is an independent
appraisal or an investor offer. The numerical results do not estimate the odds
of platform success. No customer interest has been reported.

The founder instructed on 3 October 2026 that the repository is now private.
These are local private-review artifacts. Distribution, partner outreach and
future progress publication require a deliberate disclosure decision; building
the files does not authorize those actions. Earlier public disclosures remain
part of IP diligence.

- [Editable slide narrative and source registry](PITCH_DECK.json)
- [Slide-by-slide speaker and diligence notes](PITCH_DECK_NOTES.md)
- [Investor-reading review and rationale for the revision](INVESTOR_READING_REVIEW.md)
- [Frozen evidence hashes, derived metrics, budget and financial assumptions](pitch_deck_evidence_20261003.json)
- [Complete same-unit benchmark ledger](pitch_deck_benchmarks.csv)
- [Valuation sensitivity calculations](pitch_deck_financial_sensitivity.csv)
- [Detailed research status report](../report/sleeping_machines_status.pdf)
- [Private diligence bundle: both PDFs, notes, CSVs, report and completed parents](sleeping_machines_private_diligence_20261003T182500Z.zip)

The older investment memo and one-page pitch retain the earlier $10M discussion
position for historical continuity. Use the new full deck for the current
financing proposal. Changing the proposal does not create new benchmark evidence.

The deck includes public founder information, AMD's Silo AI acquisition and
LUMI/Poro/Viking work, AMD/Liquid AI collaboration, and potential Intel/Google
strategic fit. Precedents do not establish interest in this project or provide
direct pre-seed valuation comparables. Sole-founder status does not establish
sole research authorship or exclusive IP ownership: preserve Karoliina Salminen's
existing research credit and resolve contributions and employer assignments.

The investor-reading revision uses the same frozen research evidence. Internal
configurations are translated into readable labels and defined in the appendix.
It introduces a specific product hypothesis, annual per-customer economics,
three commercial proof gates and a budget tied to those gates. Conditional exit
arithmetic is retained in the appendix rather than carrying the main pitch.
Previous PDFs, source snapshots and diligence bundles remain in the historical
record; use the current links above for review.

To deliberately refresh the frozen evidence and render with bounded resources:

```bash
python3 scripts/build_pitch_deck.py --prepare
.venv-docker/bin/python scripts/build_pitch_deck.py --tag UNIQUE_PUBLICATION_TAG
```

Preparation uses the existing completed evidence snapshot. It does not discover
new experiment results automatically. The renderer uses one CPU thread, nice19,
a 300,000 KiB RSS watchdog, 1,000,000 KiB address-space cap, 120-second timeout
and an 8 GiB available-memory floor. It imports no Torch or NumPy and executes no
model runtime. Source/evidence/output hashes guard concurrent publication; prior
deck versions and immutable publication records are retained. PDF page bounds,
pagination and source links are checked; inspect the preview visually as well.
