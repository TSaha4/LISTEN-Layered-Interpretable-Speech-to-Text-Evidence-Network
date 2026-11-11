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

## 5. Evidence of technical effect (to collect before filing)

Run `evaluate.py` on QMSum/MeetingQA-style questions with synthetic noise injected into the audio (or word-probability perturbation) and report, with gating on vs off:
- evidence precision@k and answer accuracy under noise,
- Expected Calibration Error of the reliability score (should fall after `calibrate()`),
- fraction of wrong answers flagged "low", fraction of correct answers flagged "high",
- commitment-supersession accuracy on a hand-labelled set of multi-meeting decisions.

A measurable gain in any of these is the "technical improvement" a software claim needs in India (s.3(k)) and the US (Alice).

## 6. Prior art considered

HGN (arXiv 1911.03631) · GRGA (ACL Findings 2026) · GETALP@AutoMin 2025 · MeeQA / MeetingQA · RAGONITE (arXiv 2412.10571) · RAGE (arXiv 2405.13000) · XGRAG (arXiv 2604.24623) · KGRAG-Ex (arXiv 2507.08443) · SodaMem (arXiv 2608.08055) · SCoP (arXiv 2609.22213) · US 11,640,418 B2 · uncertainty-aware representations for spoken QA (confusion-network embeddings) · spoken-document retrieval patents that drop low-confidence index terms.

None propagates word-level speech reliability through graph construction, GNN scoring, calibrated answer scoring, word-level warning and commitment supersession as one chain.
