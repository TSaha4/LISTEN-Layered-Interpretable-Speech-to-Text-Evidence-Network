# Invention Disclosure — LISTEN-U

**Working title:** Uncertainty-propagated evidence graph for question answering over spoken meetings, with calibrated answer reliability and cross-meeting commitment supersession.

**Inventor(s):** _fill in_ · **Date of this record:** 2026-11-11 · **Repository:** `TSaha4/LISTEN-Layered-Interpretable-Speech-to-Text-Evidence-Network` (branch `claude/project-patentability-assessment-6240a0`)

> This is an engineering disclosure, not legal advice. Have a registered patent agent review it before filing. Note that the base LISTEN system has been public since July 2026; the features below were added on 2026-11-11 and are what any claim must rest on. In India and Europe, file **before** publishing or demoing these features. In the US, the 12-month grace period for these new features runs from their first public disclosure.

---

## 1. Problem

Meeting question-answering systems treat the ASR transcript as ground truth. Two kinds of uncertainty are thrown away:

1. **Recognition uncertainty** — the recogniser's per-word probability that it heard the word correctly.
2. **Speaker uncertainty** — whether the person sounded sure (fillers, hedges, pauses, self-repairs).

Published work (HGN, GRGA, GETALP/AutoMin, RAGONITE, XGRAG) builds evidence graphs over text and explains answers by perturbing text. None of it carries acoustic/recognition uncertainty into the graph, into the ranking, or into what the user is shown. Separately, temporal knowledge-graph work (SodaMem, SCoP, ConTExT) tracks fact supersession but is not driven by speech reliability and is not applied to meeting commitments.

## 2. Summary of the invention

A pipeline in which uncertainty measured at the **word level of speech** is propagated, without being collapsed early, through four later stages:

| Stage | What the uncertainty does | Code |
|---|---|---|
| A. Segment annotation | Per-word ASR probability + hesitation cues → segment `reliability` | `app/slp/uncertainty.py`, `app/slp/asr.py` |
| B. Evidence graph | Reliability stored on nodes; **entity edges gated** by ASR confidence of the shared entity's words; edge confidence = weakest endpoint | `app/dl/graph_builder.py` |
| C. Graph scoring | GAT output passed through a **learnable multiplicative reliability gate** (`score · r^β`); raw and gated scores both kept | `app/dl/gat_model.py::ReliabilityGate`, `app/dl/inference.py` |
| D. Answer reliability | Calibrated logistic fusion of evidence reliability, retrieval margin, evidence agreement and grounding verdict; the low-confidence words the answer **leans on** (ASR prob < τ **and** positive SHAP) are surfaced with timestamps | `app/xai/reliability.py` |
| E. Commitment graph | Decisions/promises extracted with provenance and reliability; later conflicting statements **supersede** earlier ones **unless they are markedly less reliable**, in which case they only *dispute* | `app/commitments/extractor.py`, `app/commitments/graph.py` |

The novelty is the *chain*: the same word-level reliability signal (i) changes which graph edges exist, (ii) changes node ranking, (iii) produces a probability-calibrated answer score, (iv) picks out the exact uncertain words behind the answer, and (v) decides whether a later statement is allowed to overturn an earlier decision.

## 3. Detailed description

### 3.1 Word-level uncertainty cues (Stage A)
- ASR is run with word timestamps; each word carries `(start, end, probability)`.
- `asr_confidence` = mean word probability; `min_word_confidence` = weakest word; `no_speech_prob` from the segment.
- `hesitation_score ∈ [0,1]` from counts per ten words of fillers (`um`, `uh`…), hedge phrases (`I think`, `maybe`…), intra-segment pauses > 0.6 s, and immediate repetitions, with configurable weights.
- `evidence_reliability = (0.7·mean + 0.3·min) · (1 − no_speech) · (1 − 0.5·hesitation)`, floored at 0.05. Multiplicative so evidence is only as reliable as its weakest layer.
- `entity_confidence(segment, entity)` aligns the entity's tokens to the word stream and returns the mean probability of the matched span.

### 3.2 Uncertainty-gated evidence graph (Stage B)
- Nodes: question + retrieved segments; node 0 reliability = 1.
- Entity edge between segments *i, j* is created only if some shared entity has `min(entity_confidence_i, entity_confidence_j) ≥ 0.40`. Count of rejected links is reported (`gated_entity_edges`).
- Edge confidence: `min(r_i, r_j, best shared-entity confidence)` for entity edges; `min(r_i, r_j)` for similarity and question edges.
- Tensors `node_reliability` and `edge_confidence` travel with the PyG `Data` object.

### 3.3 Reliability gate on GAT scores (Stage C)
- `ReliabilityGate` holds two learnable scalars `β_node`, `β_edge`.
- `gated_node = σ(logit) · r^β_node`, `gated_edge = σ(logit) · c^β_edge`.
- At β = 0 the gate is the identity, so the pretrained checkpoint is a strict special case; β can be fine-tuned end-to-end. Gate weights, if trained, are stored under `gate_state_dict` in the checkpoint.
- Top-k evidence is chosen from gated scores; raw scores are retained for inspection (`raw_node_scores`).

### 3.4 Calibrated answer reliability and uncertain-word attribution (Stage D)
- Components: GAT-weighted mean reliability of the answer context; retrieval margin (top-3 vs rest); mean pairwise cosine agreement among the used segments; LLM grounding verdict.
- Logistic fusion with weights loadable from `reliability_calibration.json`; `calibrate()` re-fits them from labelled answers so the score is a probability of correctness. An unsupported answer is capped below "medium".
- `uncertain_words_in_evidence`: words with ASR probability < 0.55 **and** positive SHAP contribution, with timestamps, ranked by influence. A warning is produced when any exist.
- The LLM prompt receives per-segment reliability and flags (`transcription uncertain`, `speaker hesitant`) and is instructed to hedge accordingly.

### 3.5 Reliability-aware commitment supersession (Stage E)
- Pattern extraction of decisions, commitments and deadlines; owner (PERSON), topic entities, values (DATE/MONEY/CARDINAL…), polarity (negation).
- Statements below reliability 0.20 are not recorded.
- Topic match: embedding cosine + bonus for shared topic entities ≥ 0.60.
- Conflict = polarity flip, value change, or owner change. The later statement **supersedes** unless `later.reliability + 0.25 < earlier.reliability`, in which case it is linked as **disputes** and the earlier decision stays active.
- `timeline(topic)` returns the current decision, its history, the relations between them and the reliability of the current record.

## 4. Draft claims (for discussion with an agent)

1. A computer-implemented method for answering a question about a spoken meeting, comprising: obtaining, for each word of an ASR transcript, a recognition probability; computing for each transcript segment a reliability value from the recognition probabilities and at least one speaker-hesitation cue; constructing an evidence graph whose nodes are segments and whose edges include entity edges, **wherein an entity edge between two segments is created only when the recognition probability of the shared entity's words in both segments exceeds a threshold**; scoring nodes with a graph neural network; **modifying each node score by a function of the node's reliability value**; and generating an answer from the highest modified scores.
2. The method of claim 1, wherein the function is `score · reliability^β` and β is a learned parameter.
3. The method of claim 1, further comprising computing an answer reliability by a calibrated combination of (a) the reliability of the segments used, (b) a retrieval margin, (c) agreement among the used segments and (d) a grounding verdict.
4. The method of claim 3, further comprising identifying words in the used segments whose recognition probability is below a threshold **and** whose attribution score to the retrieval is positive, and presenting those words with their timestamps as the basis of a reliability warning.
5. The method of claim 1, further comprising extracting commitment statements from segments, each carrying the segment's reliability value; linking statements about the same topic across meetings; and marking an earlier statement superseded by a later conflicting statement **only if the later statement's reliability is not lower than the earlier statement's reliability by more than a margin**.
6. A system / non-transitory medium performing the above.

## 5. Evidence of technical effect

A measurable gain is the "technical improvement" a software claim needs in India (s.3(k)) and the US (Alice). Scripts live in `scripts/`; all numbers below are reproducible from the committed data.

### 5.1 Commitment supersession (Stage E) — `scripts/evaluate_commitments.py`

12 hand-labelled multi-meeting decision sequences (`scripts/commitment_eval_cases.json`): date/value/owner changes, negated reversals, three-step chains, unrelated topics, and the same changes delivered as hedged or badly-heard speech.

| rule | relation accuracy | active-decision accuracy |
|---|---|---|
| **LISTEN-U** (later wins unless ≥ 0.25 less reliable) | **12/12** | **12/12** |
| latest-wins (standard temporal supersession) | 8/12 | 8/12 |

Every failure of latest-wins is a case where a mumbled or hedged later statement overturned a clearly-heard decision. Caveat: author-written cases; a labelled set from real meetings should be added.

### 5.2 Evidence selection under acoustic noise (Stages A–C) — `scripts/make_noisy_transcripts.py`, `scripts/evaluate_uncertainty.py`, `scripts/fit_reliability_gate.py`

Protocol: AMI meeting ES2002a (300 segments, 6 labelled queries with gold evidence ids). Each gold segment span is re-transcribed with word timestamps from the raw audio mixed with white noise at 10, 5 and 0 dB SNR, so segment ids and gold labels stay fixed while text and word probabilities come from genuinely degraded audio. Evidence F1@5 is reported with the gate off (β = 0, original LISTEN) and at several β, then with β learned by `fit_reliability_gate.py` (GAT frozen).

**Evidence F1@5, 6 queries, ES2002a** (raw outputs in `documents/results/uncertainty_eval.json`):

| condition | mean word conf. | reliability: gold segs | reliability: non-gold | retrieval only | **gate off (β=0)** | β=0.5 | **β=0.785 (learned)** | β=1 |
|---|---|---|---|---|---|---|---|---|
| clean | 0.636 | 0.527 | 0.526 | 0.104 | 0.133 | 0.140 | **0.140** | 0.140 |
| 10 dB SNR | 0.428 | 0.459 | 0.363 | 0.053 | 0.084 | 0.124 | **0.124** | 0.124 |
| 5 dB SNR | 0.352 | 0.349 | 0.274 | 0.068 | 0.087 | 0.098 | **0.117** | 0.114 |
| 0 dB SNR | 0.256 | 0.230 | 0.218 | 0.046 | 0.099 | 0.084 | 0.053 | 0.053 |
| mean | | | | 0.068 | 0.101 | 0.111 | 0.108 | 0.108 |

Reading:
- **The reliability signal separates gold from non-gold evidence under moderate noise** (0.459 vs 0.363 at 10 dB; 0.349 vs 0.274 at 5 dB) and, as expected, not on clean audio (0.527 vs 0.526).
- **The learned gate (β = 0.785) raises evidence F1 by +48 % at 10 dB and +34 % at 5 dB, with no cost on clean audio** (+5 %). That is the technical effect: the same model, same transcript, better evidence selection when the audio is degraded.
- **At 0 dB the gate hurts** (0.099 → 0.053). The gap between gold and non-gold reliability collapses (0.230 vs 0.218): when everything is mis-heard, word probability no longer carries relevance information and the gate merely injects noise. This bounds the claim honestly — the effect holds while the recogniser still has a usable signal, and the system should fall back to β = 0 when mean segment confidence drops below ≈ 0.3 (a dependent claim worth adding: *adaptive β from mean transcript confidence*).
- β was learned with a pairwise ranking loss (`fit_reliability_gate.py`), GAT frozen, pooled over all four conditions; the fitted value is stored in the checkpoint as `gate_state_dict` and picked up by `GATInference`. A BCE objective was tried first and drives β → 0 (shrinking probabilities always lowers the likelihood of positives), which is why ranking loss is the correct objective for a re-ranking gate.
- Absolute F1 is low for every variant because these are QMSum summarisation queries with ≈ 8 gold segments each, so top-5 cannot exceed ≈ 0.77 and the retrieval stage itself scores 0.05–0.10. Relative gain over the ungated model is the comparison that matters.
- Caveat: single meeting, six queries, white noise only, and β was fitted on this same meeting, so the gains above are in-sample. The held-out replication on six more meetings is in §5.5 and does **not** reproduce them.

### 5.3 Calibration of the answer-reliability score (Stage D)

Leave-one-query-out fit of the logistic weights over all 24 (query × condition) cases; label = gated top-5 evidence F1 ≥ 0.15 (proxy for "answer is correct"; no LLM call).

| gate | positive rate | ECE, default weights | ECE, LOO-fitted | Brier, default | Brier, LOO-fitted |
|---|---|---|---|---|---|
| β = 0 | 0.38 | 0.563 | **0.417** | 0.560 | **0.298** |
| β = 0.5 | 0.42 | 0.531 | **0.358** | 0.526 | **0.302** |

Fitting the calibration weights cuts ECE by roughly a third and Brier score by almost half, i.e. the reliability number moves from "overconfident heuristic" towards a usable probability. Caveat: proxy label, 24 cases; needs real answer-correctness labels with an LLM in the loop.

### 5.4 Still to collect

- Fraction of wrong answers flagged "low" / correct flagged "high" with a real LLM in the loop.
- A direct test of Stages A–B that does not go through top-5 F1: AUROC of segment reliability for gold vs non-gold candidates per condition, pooled over meetings (the F1@5 test in §5.5 is near its floor and leaves the top-5 unchanged on 29 of 40 queries).
- Re-run §5.5 with β fitted leave-one-meeting-out, and with queries restricted to compact gold sets (≤ 20 segments) where top-5 F1 can move.
- Babble / reverberant noise instead of white noise; a stronger retriever so that gating acts on candidates that contain gold at all.

### 5.5 Multi-meeting replication — `scripts/build_aligned_labels.py`, `scripts/aggregate_uncertainty_eval.py`

Protocol: the §5.2 pipeline was repeated on six further AMI meetings (ES2002b, ES2002c, ES2002d, ES2003a, ES2003b, ES2003c; 19–44 min each, 154–694 Whisper segments). Gold evidence ids were produced by `scripts/build_aligned_labels.py`, which re-implements the ES2002a alignment (MiniLM cosine between each QMSum turn and nearby Whisper segments, kept at similarity ≥ 0.4, query labels = segments matched to turns inside the QMSum `relevant_text_span`); the original alignment script was not committed, and the re-implementation recovers 114 of the 118 stored ES2002a turn matches. Each meeting was then re-transcribed at clean / 10 / 5 / 0 dB SNR and scored with the **same gate (β = 0.785) learned on ES2002a**, so the six new meetings are a held-out test. 34 new labelled queries; 40 in total with ES2002a. Raw outputs: `documents/results/<meeting>_uncertainty_eval.json`, `documents/results/<meeting>_aligned_labels.json`, full table in `documents/results/multi_meeting_eval.md`.

**Evidence F1@5, gate off (β = 0) vs learned gate (β = 0.785):**

| meeting | queries | clean β=0 | clean gate | 10 dB β=0 | 10 dB gate | 5 dB β=0 | 5 dB gate | 0 dB β=0 | 0 dB gate |
|---|---|---|---|---|---|---|---|---|---|
| ES2002a (pilot, in-sample) | 6 | 0.133 | **0.140** | 0.084 | **0.124** | 0.087 | **0.117** | 0.099 | 0.053 |
| ES2002b | 6 | 0.199 | 0.172 | 0.036 | 0.032 | 0.012 | 0.008 | 0.010 | 0.004 |
| ES2002c | 5 | 0.061 | 0.061 | 0.130 | **0.136** | 0.154 | 0.116 | 0.099 | 0.025 |
| ES2002d | 6 | 0.035 | 0.017 | 0.029 | 0.002 | 0.004 | 0.004 | 0.002 | **0.004** |
| ES2003a | 6 | 0.204 | 0.106 | 0.135 | 0.070 | 0.091 | **0.093** | 0.061 | **0.074** |
| ES2003b | 6 | 0.067 | 0.042 | 0.006 | **0.028** | 0.032 | **0.067** | 0.000 | 0.000 |
| ES2003c | 5 | 0.062 | 0.050 | 0.040 | 0.040 | 0.040 | 0.000 | 0.090 | 0.012 |
| **pooled, 40 queries** | 40 | 0.111 | 0.085 (−23 %) | 0.065 | 0.060 (−7 %) | 0.058 | 0.058 (−0 %) | 0.049 | 0.025 (−50 %) |
| paired bootstrap 95 % CI of gain | | | [−0.053, −0.002] | | [−0.028, +0.019] | | [−0.026, +0.025] | | [−0.053, +0.002] |
| meetings helped / hurt | | | 1 / 5 | | 3 / 3 | | 3 / 3 | | 2 / 4 |

**Pooled reliability of gold vs non-gold candidates:** clean 0.533 / 0.539; 10 dB 0.373 / 0.356 (gap +0.018); 5 dB 0.288 / 0.266 (gap +0.022); 0 dB 0.191 / 0.193. On ES2002a alone the gaps were +0.096 and +0.075; on ES2002d, ES2003a, ES2003b and ES2003c the gap at 10 dB is *reversed* (gold segments are less reliable than the distractors).

Reading — this is a negative replication and the disclosure should say so:
- **The pilot gain does not hold out of sample.** Pooled over 40 queries the learned gate is significantly *worse* on clean audio (−23 %, CI excludes zero), indistinguishable from the ungated model at 10 dB and 5 dB (CIs straddle zero, 3 meetings helped vs 3 hurt), and worse at 0 dB. No β among {0.5, 0.785, 1.0} beats β = 0 pooled, on any condition, by more than 0.001 F1. The +48 % / +34 % in §5.2 came from fitting β on the same six queries it was then scored on.
- **The reliability signal is only weakly informative across meetings.** The gold-vs-non-gold gap that drove the pilot (≈ +0.09) shrinks to ≈ +0.02 pooled and is reversed on four of the six new meetings at 10 dB. Word-level ASR confidence tracks acoustic degradation (mean confidence falls monotonically with SNR on every meeting) but, on these meetings, it does not track *which* candidate segments are the evidence.
- **The test has low power, which cuts both ways.** Absolute F1@5 is 0.00–0.20 for every variant; on 29 of 40 queries the gate leaves the top-5 unchanged, and QMSum gold sets are large (up to 160 segments) and only approximately aligned, so top-5 F1 is a blunt instrument. This limits how strongly the negative result can be read, but it equally means the §5.2 numbers cannot be cited as evidence of effect.
- **Implication for the claims.** The "measurable gain in evidence selection under noise" effect is not currently demonstrated. Independent claims should rest on the mechanism (Stages A–E as an integrated, auditable chain) and on the Stage D calibration and Stage E commitment results, not on §5.2. Any effect claim for Stage C should wait for the §5.4 experiments; an adaptive-β claim that falls back to β = 0 when the gold/non-gold reliability gap is absent cannot be enforced at inference time, since gold is unknown, so it does not rescue the result.

## 6. Prior art considered

HGN (arXiv 1911.03631) · GRGA (ACL Findings 2026) · GETALP@AutoMin 2025 · MeeQA / MeetingQA · RAGONITE (arXiv 2412.10571) · RAGE (arXiv 2405.13000) · XGRAG (arXiv 2604.24623) · KGRAG-Ex (arXiv 2507.08443) · SodaMem (arXiv 2608.08055) · SCoP (arXiv 2609.22213) · US 11,640,418 B2 · uncertainty-aware representations for spoken QA (confusion-network embeddings) · spoken-document retrieval patents that drop low-confidence index terms.

None propagates word-level speech reliability through graph construction, GNN scoring, calibrated answer scoring, word-level warning and commitment supersession as one chain.
