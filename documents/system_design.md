# LISTEN System Design and Workflow

This document provides a visual overview of the **LISTEN (Layered Interpretable Spoken Transcript Evidence Network)** architecture, outlining how data flows from user input through processing, reasoning, and final answer generation with explainability.

```mermaid
graph TD
    classDef input fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#01579b;
    classDef process fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100;
    classDef output fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20;
    classDef storage fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#4a148c;
    classDef highlight fill:#ffebee,stroke:#f44336,stroke-width:2px,color:#b71c1c;

    User((User)):::input

    %% Inputs
    User -- "1. Upload Meeting" --> MediaInput(Raw Audio/Video):::input
    User -- "3. Ask Question" --> QueryInput(User Query):::input

    %% Storage
    DB[(Data Store: Transcripts, Indices, Entities)]:::storage

    %% SLP Pipeline
    subgraph SLP ["Speech & Language Processing (SLP)"]
        MediaInput --> Whisper["Whisper ASR\n(Transcription)"]:::process
        Whisper --> SpaCy["spaCy NER\n(Entity Extraction)"]:::process
    end
    SpaCy -- "2. Save Processed Data" --> DB

    %% Phase 1: Retrieval
    subgraph Phase1 ["Phase 1: Retrieval"]
        QueryInput --> BiEncoder["Bi-encoder\nSimilarity Search"]:::process
        DB --> BiEncoder
        BiEncoder --> FAISS["FAISS Index"]:::storage
        FAISS --> CandidateSegments("Top Candidate Segments"):::process
    end

    %% Phase 2: Graph Reasoning
    subgraph Phase2 ["Phase 2: Graph Construction & Reasoning"]
        CandidateSegments --> GraphBuilder["Construct Evidence Graph\n(Nodes=Segments, Edges=Entities/Semantics)"]:::process
        GraphBuilder --> GAT["Graph Attention Network (GAT)\n(Multi-hop Reasoning)"]:::process
        GAT --> WeightedSegments("Top-Weighted Segments"):::process
    end

    %% Phase 3: Generation & XAI
    subgraph Phase3 ["Phase 3: Answer Generation & Explainability"]
        WeightedSegments --> LLM["Multi-provider LLM\n(Gemini / OpenAI / Groq)"]:::process
        QueryInput --> LLM
        WeightedSegments --> SHAP["XAI: SHAP Values\n(Word-level Influence)"]:::highlight
    end

    %% Outputs
    LLM --> Answer["Concise Natural-Language Answer"]:::output
    GAT --> InteractiveGraph["Interactive Physics Graph\n(Visual Evidence Paths)"]:::output
    SHAP --> EvidenceDisplay["SHAP Evidence Highlights\n(Red = Positive Influence)"]:::output

    %% UI Delivery
    Answer --> UI["Frontend UI"]:::input
    InteractiveGraph --> UI
    EvidenceDisplay --> UI
    UI -. "Presents to" .-> User
```

### Flow Breakdown:
1. **Speech & Language Processing (SLP):** The system first transcribes raw media uploads using Whisper and extracts named entities using spaCy. This processed text data is indexed and stored.
2. **Phase 1 (Retrieval):** When a user submits a query, a bi-encoder similarity search alongside FAISS quickly retrieves the most relevant segments from the transcript.
3. **Phase 2 (Graph Construction & GAT Reasoning):** The candidate segments are assembled into a graph (nodes are segments, edges are shared entities/semantic meaning). A trained Graph Attention Network (GAT) traverses this graph to perform multi-hop reasoning, ultimately assigning weights to segments based on their relevance.
4. **Phase 3 (Answer Generation):** The highest-weighted segments are passed as context to an LLM to generate a natural language response to the user's question.
5. **Explainable AI (XAI):** Simultaneously, SHAP values are computed on the selected evidence, scoring exactly which words positively influenced the model's retrieval (highlighted in red on the frontend).
6. **Frontend Delivery:** The final outputs—the LLM Answer, the Interactive Evidence Graph, and the SHAP highlights—are all presented to the user via the frontend UI.
