# Dataset Collection and Justification

The LISTEN system utilizes two primary datasets for training its Graph Attention Network (GAT).

### 1. HotpotQA Dataset
*   **Split Used**: `distractor` split (via HuggingFace).
*   **Purpose**: Initial Pre-training.
*   **Justification**: HotpotQA is a dataset designed specifically for multi-hop question answering. In the "distractor" setting, a question is provided alongside multiple paragraphs (some relevant, some distractors). The LISTEN system uses this to train the GAT to perform multi-hop reasoning across a graph of text segments. It teaches the model how to traverse nodes (sentences/segments), follow entity/semantic connections, and correctly identify the supporting facts while ignoring irrelevant information. 

### 2. QMSum Dataset
*   **Source**: Yale-LILY QMSum meeting dataset.
*   **Purpose**: Domain-Specific Fine-Tuning.
*   **Justification**: While HotpotQA teaches the network general multi-hop reasoning, it consists of clean Wikipedia text. The LISTEN system is designed to handle *spoken meeting transcripts*. QMSum contains real meeting transcripts paired with both general and specific queries and their corresponding relevant text spans. Fine-tuning on QMSum adapts the model's reasoning capabilities to the messy, conversational domain of spoken meetings, ensuring it can accurately retrieve evidence from actual meeting data.
