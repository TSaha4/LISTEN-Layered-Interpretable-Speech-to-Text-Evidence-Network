# LISTEN: Deep Learning Architecture, Mathematical Formulations, & Defense Guide

> **Project Name:** **LISTEN** (*Layered Interpretable Spoken Transcript Evidence Network*)  
> **Target Audience:** Review Faculty & Defense Committee  
> **Focus:** Hard Deep Learning, Graph Neural Networks, Acoustic Preprocessing, Mathematical Formulations, Ablations, and Academic Comparison against Base Paper (*Microsoft HGN*).

---

## Quick Navigation Index
1. [Acronyms & Full Forms (Sabhi Full Forms)](#1-acronyms--full-forms)
2. [Datasets Used: Sources, Splits & Technical Justifications](#2-datasets-used)
3. [Model Selections: Why These Models? (Scratch vs. Foundation Models)](#3-model-selections--why-these-models)
4. [LISTEN vs. Base Paper (Microsoft HGN) — In-Depth Architectural Contrast](#4-listen-vs-base-paper-microsoft-hgn)
5. [The Deep Learning Pipeline (Step-by-Step Mathematical Flow)](#5-the-deep-learning-pipeline-step-by-step)
   - [Phase 0: Acoustic Frontier & Signal Preprocessing](#phase-0-acoustic-frontier--signal-preprocessing)
   - [Phase 1: Named Entity Extraction & Graph Linking (NER)](#phase-1-named-entity-extraction--graph-linking-ner)
   - [Phase 2: Dense Semantic Embedding & FAISS Candidate Filtering](#phase-2-dense-semantic-embedding--faiss-candidate-filtering)
   - [Phase 3: Heterogeneous Evidence Graph Formulation](#phase-3-heterogeneous-evidence-graph-formulation)
   - [Phase 4: Graph Attention Network (GAT) Forward Pass & Attention Formulation](#phase-4-graph-attention-network-gat-forward-pass--attention-formulation)
   - [Phase 5: Loss Function Formulation & Optimization](#phase-5-loss-function-formulation--optimization)
6. [Codebase Architecture Map (Exact Files & Line Numbers)](#6-codebase-architecture-map)
7. [Defense FAQ: "Shut the Faculty's Trap" (Brutal Cross-Examination Answers)](#7-defense-faq-brutal-cross-examination-answers)

---

## 1. Acronyms & Full Forms

| Acronym | Full Form | Meaning in Layman Terms (*Aasan Bhasha Mein*) |
| :--- | :--- | :--- |
| **LISTEN** | **L**ayered **I**nterpretable **S**poken **T**ranscript **E**vidence **N**etwork | Humare project ka naam: Audio sunkar graph banakar saboot ke saath answer dene wala DL system. |
| **HGN** | **H**ierarchical **G**raph **N**etwork | Humara base paper (Microsoft Research, Fang et al., 2020) for multi-hop QA over text. |
| **GAT** | **G**raph **A**ttention **N**etwork | Deep Learning graph architecture jisme connected nodes ek doosre ko dynamically *attention weights* ke basis par information pass karte hain. |
| **GNN** | **G**raph **N**eural **N**etwork | Family of deep neural nets jo tabular ya sequential data ke badle graphs $(\mathcal{V}, \mathcal{E})$ par kaam karte hain. |
| **ASR** | **A**utomatic **S**peech **R**ecognition | Audio sound waves ko written text sentences mein convert karne wali AI technology. |
| **VAD** | **V**oice **A**ctivity **D**etection | Audio mein se silence aur background noise kaat kar sirf insani aawaz (speech) ko isolate karna. |
| **PCM** | **P**ulse **C**ode **M**odulation | Uncompressed raw digital audio format (16-bit integers, 16,000 samples per second). |
| **NER** | **N**amed **E**ntity **R**ecognition | Text mein se specific entities (insan ka naam, company, date, time) ko dhoondh kar classify karna. |
| **FAISS** | **F**acebook **A**I **S**imilarity **S**earch | High-dimensional vector embeddings mein fastest $O(1) / O(\log N)$ nearest-neighbor cosine search library. |
| **XAI** | e**X**plainable **A**rtificial **I**ntelligence | Black-box model ko open karke mathematically dikhana ki model ne ye decision kyu liya. |
| **SHAP** | **SH**apley **A**dditive ex**P**lanations | Game theory par based mathematical algorithm jo har input word ka contribution score nikalta hai. |
| **BCE** | **B**inary **C**ross-**E**ntropy | Loss function jo binary classification (relevance: 0 ya 1) ke liye probability error calculate karta hai. |

---

## 2. Datasets Used

Humne project mein 2 primary benchmark datasets use kiye hain:

### A. HotpotQA Dataset (Yang et al., EMNLP 2018)
* **Dataset Availability:** Publicly hosted on HuggingFace Datasets (`datasets.load_dataset("hotpot_qa", "distractor")`) and official site [https://hotpotqa.github.io/](https://hotpotqa.github.io/).
* **Split Used:** `distractor` split (90,447 training pairs, 7,405 dev pairs).
* **Structure:** Har question ke saath 10 paragraphs hote hain (2 gold paragraphs jisme answer chupa hai + 8 hard distractors). Isme question answer karne ke liye *multi-hop reasoning* (Paragraph 1 $\to$ Bridge Entity $\to$ Paragraph 2) mandatory hoti hai.
* **Why this dataset?**  
  *GAT ko multi-hop reasoning sikhane ke liye.* Single-hop dataset (jaise SQuAD) me graph ki zaroorat nahi padti. HotpotQA forces the network to learn graph hops between entities and sentences to find the ground-truth supporting facts.

### B. QMSum Dataset (Zhong et al., NAACL 2021 / Yale-LILY Lab)
* **Dataset Availability:** GitHub repo `Yale-LILY/QMSum` and HuggingFace.
* **Split Used:** Product meetings & Academic committee meeting transcripts (`data/qmsum/`).
* **Structure:** Real-world spoken meetings (AMI & ICSI corpora) with multi-turn dialogues, interruptions, informal slang, and paired queries with *relevant turn spans*.
* **Why this dataset?**  
  *Domain Adaptation.* HotpotQA ka text Wikipedia ka hai — bilkul clean aur formal. Lekin humara LISTEN system spoken meetings aur speech par chalta hai. Agar model ko sirf HotpotQA par train karte, toh real meeting mein "uh, umm, yeah, right" dekh kar GAT fail ho jata. QMSum se fine-tune karke GAT ne conversational evidence spans dhoondhna seekha.

---

## 3. Model Selections — Why These Models?

> **Faculty Question:** *"Whisper aur MiniLM download karke laga diya... khud kyu nahi banaya model from scratch?"*

### 1. Acoustic Speech Model: `faster-whisper` (OpenAI Whisper base)
* **Available at:** HuggingFace / CTranslate2 model hub (`Systran/faster-whisper-base`).
* **Why Whisper and not scratch CNN-BiLSTM-CTC?**  
  *Faculty ko seedha bolna:* Speech recognition from scratch train karne ke liye **680,000 ghante ka labeled multi-speaker audio data** aur 100+ A100 GPUs chahiye hote hain ($100,000+ cost). Undergraduate project ka scope speech recognition engine invent karna nahi hai; humara contribution **Downstream Graph-based Spoken Transcript Reasoning** hai. Whisper provides state-of-the-art robust phoneme recognition even under acoustic reverb.
* **Why `faster-whisper` instead of vanilla OpenAI Whisper?**  
  Vanilla Whisper runs on standard PyTorch fp32 ($4\times$ memory, slow CPU decoding). `faster-whisper` uses **CTranslate2 (int8 quantization)** with Silero VAD, cutting inference latency by **$4\times$** on CPU with zero loss in word error rate (WER).

### 2. Dense Semantic Embedder: `sentence-transformers/all-MiniLM-L6-v2`
* **Available at:** HuggingFace Hub (`sentence-transformers/all-MiniLM-L6-v2`).
* **Why this encoder?**  
  6-layer distilled Transformer with **384-dimensional embeddings**. Full BERT-large is 1024-D and has 340M parameters. MiniLM gives 98% of BERT's semantic ranking accuracy at $5\times$ speed and produces normalized dense representations perfect for inner-product FAISS retrieval.

### 3. Named Entity Recognizer: `spaCy` (`en_core_web_sm`)
* **Available at:** Explosion AI model repository (`python -m spacy download en_core_web_sm`).
* **Why spaCy?**  
  It provides ultra-low latency token-level Transition-Based Named Entity Recognition with pre-trained word vectors, identifying PERSON, ORG, DATE, and GPE tags in sub-millisecond time.

### 4. Custom Deep Learning Model (OUR CONTRIBUTION): `EvidenceGAT`
* **Built From Scratch In:** [app/dl/gat_model.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/gat_model.py) and [phase2_graph/train.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/phase2_graph/train.py).
* **Trained Weights:** `data/checkpoints/gat_qmsum.pt` (107,650 trainable parameters).
* **What is novel here?**  
  Ye koi pre-trained model nahi hai! Isko humne **PyTorch Geometric** me design kiya, architecture configure kiya, loss formulate kiya, aur QMSum/HotpotQA par **BCEWithLogitsLoss** ke saath scratch se train kiya hai.

---

## 4. LISTEN vs. Base Paper (Microsoft HGN)

Hamare base paper ka citation:
> **Yuwei Fang, Siqi Sun, Zhe Gan, Rohit Pillai, Shuohang Wang, Jingjing Liu (Microsoft Dynamics 365 AI Research)**  
> *"Hierarchical Graph Network for Multi-hop Question Answering"* — **EMNLP / arXiv:1911.03631v4 (Oct 2020)**.

### Comprehensive Comparison Matrix

| Feature / Dimension | Microsoft HGN (Base Paper) | LISTEN (Our Framework) | Technical Rationale (*Kyu Change Kiya?*) |
| :--- | :--- | :--- | :--- |
| **Input Modality** | Clean, curated Wikipedia articles | Continuous raw audio/video files (MP4/WAV) | Base paper assumes perfect text already exists. In real life, meetings are spoken audio. |
| **Acoustic Preprocessing** | *None* (Pure NLP) | 16 kHz mono PCM extraction $\to$ VAD $\to$ CTC/Attention ASR | Speech has background noise and overlapping speech that must be segmented temporally. |
| **Graph Hierarchy** | 4 Heterogeneous Tiers: $Q$ (Question), $P$ (Paragraphs), $S$ (Sentences), $E$ (Entities) | Star-and-Mesh Evidence Graph: $Q$ (Question Node) + $\{S_1, \dots, S_K\}$ (Retrieved Segments) | Meeting transcripts have no notion of "Wikipedia Paragraphs". Spoken discourse is turn-based. |
| **Edge Definitions** | 7 edge types (including Wikipedia Hyperlinks) | 3 relation types: Type 0 ($Q \leftrightarrow S$), Type 1 (Shared Entity), Type 2 (Semantic Cosine $\ge 0.75$) | Wikipedia has hardcoded hyperlinks ($P_1 \to P_2$). Spoken conversations have NO hyperlinks, so we dynamically generate edges using NER overlap and semantic cosine similarity! |
| **Graph Scale / Memory** | Fixed padding: $n_p=4, n_s=40, n_e=60 \implies 105$ nodes per example | Dynamic candidate subgraph: $|V| = K + 1 = 16$ nodes, $|E| \approx 50-70$ edges | Avoids quadratic memory footprint and prevents GNN over-smoothing. |
| **Backbone Encoder** | RoBERTa-large (355M params) + BiLSTM (1.44M params) | `all-MiniLM-L6-v2` (22M params) + GATConv | RoBERTa-large requires 4 Quadro RTX 8000 GPUs (64 GB VRAM) for 12 hours. MiniLM + GAT trains efficiently on consumer hardware. |
| **Loss Objective** | Multi-task loss: $\mathcal{L}_{start} + \mathcal{L}_{end} + \lambda_1 \mathcal{L}_{para} + \lambda_2 \mathcal{L}_{sent} + \lambda_3 \mathcal{L}_{ent} + \lambda_4 \mathcal{L}_{type}$ | Multi-hop Evidence Selection BCEWithLogitsLoss: $\mathcal{L}_{node} = \text{BCE}(\hat{y}_{node}, y) + \lambda \mathcal{L}_{edge}$ | We isolate the evidence reasoning component to feed verified evidence to downstream LLMs or humans. |
| **Explainability (XAI)** | Raw attention visualization | Dual-level XAI: Learned GAT Attention $\alpha_{ij}$ + Token-level Kernel SHAP attributions | Base paper leaves span extraction to black-box gated attention. We provide mathematical token-level proof of attribution. |

---

## 5. The Deep Learning Pipeline (Step-by-Step)

```
[Raw Video/Audio (MP4/WAV)] 
             │
             ▼ (ffmpeg: 16 kHz mono PCM)
     [Audio Waveform]
             │
             ▼ (faster-whisper + Silero VAD)
  [Transcript Segments with Timestamps]
             │
             ▼ (spaCy en_core_web_sm)
   [Named Entity Mentions per Segment]
             │
             ▼ (all-MiniLM-L6-v2 Embedder)
  [384-D Normalized Dense Embeddings]
             │
             ▼ (FAISS IndexFlatIP Cosine Search)
   [Top-K Candidate Segments (K=15)]
             │
             ▼ (EvidenceGraphBuilder: 3 Edge Types)
[PyTorch Geometric Data(x, edge_index, edge_type)]
             │
             ▼ (EvidenceGAT: 2 GATConv Layers + Multi-Head Attention)
[Node Scores σ(logits) & Edge Attention Weights α_ij]
```

---

### Phase 0: Acoustic Frontier & Signal Preprocessing
1. **Extraction:** `ffmpeg` extracts the audio stream from the uploaded MP4:
   $$\text{Command: } \texttt{ffmpeg -i input.mp4 -vn -acodec pcm\_s16le -ar 16000 -ac 1 output.wav}$$
   *Meaning in layman terms:* Video stream ko drop karo (`-vn`), 16-bit uncompressed audio lo (`pcm_s16le`), sample rate ko 16,000 samples per second set karo (`-ar 16000`), aur stereo ko mono single-channel banao (`-ac 1`).
2. **ASR Decoding:** `faster-whisper` decodes the 16 kHz waveform into timestamped segments:
   $$S = \{s_1, s_2, \dots, s_M\}, \quad s_i = (\text{text}_i, t_{start}^{(i)}, t_{end}^{(i)})$$

---

### Phase 1: Named Entity Extraction & Graph Linking (NER)
1. **Surface Form Extraction:** For each segment text, spaCy extracts entities:
   $$\mathcal{E}(s_i) = \{e_{i,1}, e_{i,2}, \dots, e_{i,p}\} \quad \text{where } e \in \{\text{PERSON, ORG, DATE, TIME, GPE}\}$$
2. **Shared Entity Logic:** If segment $s_i$ and segment $s_j$ contain a common entity:
   $$\mathcal{E}(s_i) \cap \mathcal{E}(s_j) \neq \emptyset \implies \text{Create Edge Type 1 between } s_i \text{ and } s_j$$
   *Layman:* Agar ek segment me Steve Jobs ne bola "Reed College" aur 5 minute baad doosre segment me bola "Reed College", toh graph dono segments ke beech me seedha rasta (edge) bana deta hai!

---

### Phase 2: Dense Semantic Embedding & FAISS Candidate Filtering
1. **Vector Projection:** We map question text $q$ and segment texts $s_i$ into $\mathbb{R}^{384}$:
   $$h_q = f_{\theta}(q) \in \mathbb{R}^{384}, \quad h_{s_i} = f_{\theta}(s_i) \in \mathbb{R}^{384}$$
   Vectors are $L_2$-normalized: $\|h_q\|_2 = 1, \quad \|h_{s_i}\|_2 = 1$.
2. **Inner Product Search (FAISS):**
   $$\text{Cosine Similarity}(q, s_i) = h_q \cdot h_{s_i}^T$$
   We select the Top-$K$ candidates ($K=15$) to construct the localized evidence subgraph.

---

### Phase 3: Heterogeneous Evidence Graph Formulation
We construct a PyTorch Geometric `Data` object:
* **Node Feature Matrix $X \in \mathbb{R}^{(K+1) \times 384}$:**
  $$X = \begin{bmatrix} h_q \\ h_{s_1} \\ \vdots \\ h_{s_K} \end{bmatrix}$$
  Row $0$ is always the Question Node. Rows $1 \dots K$ are candidate segment nodes.
* **Edge Index Matrix $\mathcal{E} \in \mathbb{Z}^{2 \times |E|}$ with 3 Relation Types:**
  * **Type 0 (Question Links):** Bidirectional edges $(0 \leftrightarrow i)$ for all $i \in \{1 \dots K\}$.
  * **Type 1 (Shared Entity Links):** Bidirectional edges $(i \leftrightarrow j)$ if $\text{Entities}(s_i) \cap \text{Entities}(s_j) \neq \emptyset$.
  * **Type 2 (Semantic Proximity Links):** Bidirectional edges $(i \leftrightarrow j)$ if $\cos(h_{s_i}, h_{s_j}) \ge 0.75$.

---

### Phase 4: Graph Attention Network (GAT) Forward Pass & Attention Formulation
Our model has:
* **Linear Input Projection:** $\mathbb{R}^{384} \to \mathbb{R}^{128}$.
* **Layer 1 GATConv:** 4 attention heads ($d_{head} = 32$, total $= 128$).
* **Layer 2 GATConv:** 1 attention head ($d = 128$).
* **Scoring Heads:** 2-layer MLPs for Node relevance and Edge relevance.

#### Mathematical Equation for Multi-Head Attention:
For node $i$ and neighbor $j \in \mathcal{N}_i$ under attention head $k$:
$$\alpha_{ij}^k = \frac{\exp\left( \text{LeakyReLU}\left( \mathbf{a}_k^T [W_k h_i \,\|\, W_k h_j] \right) \right)}{\sum_{l \in \mathcal{N}_i} \exp\left( \text{LeakyReLU}\left( \mathbf{a}_k^T [W_k h_i \,\|\, W_k h_l] \right) \right)}$$
Where:
* $W_k \in \mathbb{R}^{d_{out} \times d_{in}}$ is the learned linear transformation weight matrix.
* $[\cdot \,\|\, \cdot]$ denotes vector concatenation ($\mathbb{R}^{2d_{out}}$).
* $\mathbf{a}_k \in \mathbb{R}^{2d_{out}}$ is the learned attention parameter vector.
* $\alpha_{ij}^k$ is the normalized attention coefficient (scalar weight).

The aggregated hidden state at Layer 1 is:
$$h_i^{(1)} = \Vert_{k=1}^4 \sigma\left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W_k h_j^{(0)} \right)$$

#### Output Scoring:
$$o_{node} = \text{MLP}_{node}(h_i^{(2)}) = W_2 \cdot \text{ReLU}(W_1 h_i^{(2)} + b_1) + b_2 \in \mathbb{R}^1$$
The probability that segment $i$ is ground-truth supporting evidence is:
$$p_i = \sigma(o_{node}^{(i)}) = \frac{1}{1 + e^{-o_{node}^{(i)}}}$$

---

### Phase 5: Loss Function Formulation & Optimization
During training ([phase2_graph/train.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/phase2_graph/train.py)), question nodes ($i=0$) are masked out. For all segment nodes $i \in \{1 \dots K\}$ with ground truth binary label $y_i \in \{0, 1\}$:

$$\mathcal{L}_{BCE} = - \frac{1}{K} \sum_{i=1}^K \left[ y_i \log \sigma(o_i) + (1 - y_i) \log (1 - \sigma(o_i)) \right]$$

To avoid numerical instability, we use PyTorch's `nn.BCEWithLogitsLoss`, which integrates the sigmoid into the log formula via the log-sum-exp trick:
$$\ell(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

* **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, Weight Decay $= 10^{-4}$).
* **Learning Rate:** $10^{-4}$ with Linear warmup.
* **Gradient Clipping:** Max norm $= 1.0$ to prevent exploding gradients in deep graphs.

---

## 6. Codebase Architecture Map

| Pipeline Stage | Project Source File | Exact Line Range | Core Function / Class |
| :--- | :--- | :--- | :--- |
| **Video $\to$ Mono Audio** | [app/slp/audio_extraction.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/slp/audio_extraction.py) | Lines 11–59 | `extract_audio_from_video()` |
| **Whisper ASR + VAD** | [app/slp/asr.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/slp/asr.py) | Lines 38–76 | `transcribe_audio()` |
| **spaCy NER Extraction** | [app/slp/entity_extraction.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/slp/entity_extraction.py) | Lines 17–70 | `enrich_segments_with_entities()` |
| **Vector Embeddings (384-D)** | [app/dl/embeddings.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/embeddings.py) | Lines 22–45 | `embed_texts()` |
| **FAISS Nearest Neighbor Index**| [app/dl/embeddings.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/embeddings.py) | Lines 48–110 | `SegmentIndex.build()`, `search()` |
| **Candidate Retrieval (Top-K)** | [app/dl/retrieval.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/retrieval.py) | Lines 14–43 | `retrieve_candidates()` |
| **Graph Builder (3 Edge Types)**| [app/dl/graph_builder.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/graph_builder.py) | Lines 48–176 | `EvidenceGraphBuilder.build()` |
| **Serving GAT Neural Net** | [app/dl/gat_model.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/gat_model.py) | Lines 16–53 | `EvidenceGAT(nn.Module)` |
| **GAT Inference & Checkpoint** | [app/dl/inference.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/dl/inference.py) | Lines 15–55 | `GATInference.run()` |
| **PyG Dataset Construction** | [phase2_graph/dataset.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/phase2_graph/dataset.py) | Lines 35–180 | `GraphDataset`, `build_with_labels()` |
| **GAT Training Loop & Loss** | [phase2_graph/train.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/phase2_graph/train.py) | Lines 71–130 | `train_epoch()`, `BCEWithLogitsLoss` |
| **SHAP Token Interpretability** | [app/xai/shap_explainer.py](file:///c:/Users/sahaT/Desktop/Tanmoy/VIT/7th%20sem/project/app/xai/shap_explainer.py) | Lines 22–95 | `explain_segment()` |

---

## 7. Defense FAQ: Brutal Cross-Examination Answers

### Q1. "Graph Attention Network lagane ka fayda kya hua? Direct cosine similarity se top-5 utha lete!"
* **Technical Counter:**  
  "Sir/Ma'am, standard dense retrieval (cosine similarity) is strictly **single-hop and independent**. If a meeting discussion spans across 10 minutes where the speaker mentions a concept in Turn A, refers to it by a pronoun/alias in Turn B, and gives the actual answer in Turn C, cosine similarity fails to retrieve Turn C because Turn C doesn't have direct textual overlap with the query.  
  Our GAT executes **multi-hop message passing**. By forming edges over shared named entities and high semantic affinity, the representation of Turn C is updated by information propagating from Turn A through attention coefficient $\alpha_{AC}$. This allows the model to elevate non-obvious supporting evidence that flat vector search misses."
  *(Layman Hindi: Flat search me agar question ke keywords segment me nahi hain toh wo ignore ho jayega. GAT me shared entity ke bridge se message pass hoke wo segment bhi rank ho jata hai).*

### Q2. "Base paper (Microsoft HGN) me 4-level hierarchy thi (Q, Paragraph, Sentence, Entity). Tumne Paragraph aur Entity ko node kyu nahi banaya?"
* **Technical Counter:**  
  "Microsoft HGN was engineered for **Wikipedia articles** in HotpotQA, which have clear structural hierarchy: Document $\to$ Paragraphs $\to$ Sentences $\to$ Hyperlinked Wikipedia pages.  
  In **spoken conversational video/meetings**, there is no concept of a 'paragraph'. Speech consists of continuous temporal turns and utterances. If we created separate nodes for every single word entity, the graph would blow up to 500+ nodes for a 15-minute meeting, leading to **Laplacian over-smoothing** (where all node features become identical after 2 GNN layers). Therefore, we adopted a **Heterogeneous Star-and-Mesh topology**: entities form the *relational edges* between transcript segments rather than polluting the node space."

### Q3. "Whisper model kyu download kiya? Scratch se CNN ya RNN kyu nahi banaya ASR ke liye?"
* **Technical Counter:**  
  "Training an ASR model from scratch that handles real-world background noise, room acoustics, and accents requires a minimum of 5,000+ hours of aligned audio and massive cluster compute. Whisper was trained on 680,000 hours of multilingual weakly supervised data. Re-training an acoustic model from scratch would yield a high Word Error Rate (WER > 30%), which would propagate catastrophic errors downstream into the Graph network.  
  In modern Machine Learning research, foundation acoustic encoders are leveraged as frozen feature extractors so the research focus remains on the primary novelty: **Graph-structured Multi-Hop Evidence Reasoning**."

### Q4. "Tumhara model kaise train hua? Overfitting toh nahi ho rahi?"
* **Technical Counter:**  
  "Sir/Ma'am, our training logs are saved in `data/checkpoints/gat_train_log.json`. We trained for 20 epochs using `BCEWithLogitsLoss` with an Adam optimizer and learning rate $10^{-4}$ with gradient clipping at $1.0$.  
  To prevent overfitting:
  1. We used **Dropout ($p=0.2$)** after every GATConv layer and MLP scoring block.
  2. We added **Self-loops (`add_self_loops=True`)** to prevent central node feature attenuation.
  3. Training loss converged smoothly from **$0.966 \to 0.561$**, and validation accuracy stabilized at **$80.8\%$**."

### Q5. "Isme Explainable AI (XAI) ka kya role hai? GAT attention weights hi toh explainability hain!"
* **Technical Counter:**  
  "Recent DL literature (e.g., *'Attention is not Explanation'* by Jain & Wallace) proves that raw attention weights alone can be inconsistent indicators of feature importance.  
  Therefore, LISTEN employs a **two-tier complementary interpretability framework**:
  1. **Macro-Level (Structural Evidence):** GAT attention weights ($\alpha_{ij}$) show how information flowed between segments across the graph.
  2. **Micro-Level (Token Attribution):** Kernel SHAP computes local game-theoretic Shapley values over the individual tokens within the top-ranked segments, proving mathematically which exact spoken words caused the model's high relevance score."
