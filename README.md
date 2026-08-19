# LISTEN 

**Layered Interpretable Spoken Transcript Evidence Network**

LISTEN is an end-to-end system that takes raw audio/video recordings of meetings, transcribes them, extracts entities, and uses a Deep Learning Graph Attention Network (GAT) to answer questions based on the transcript while providing highly interpretable, explainable AI (XAI) evidence.

## Architecture

1. **Speech & Language Processing (SLP)**: Processes raw media using Whisper for transcription and spaCy for Named Entity Recognition.
2. **Phase 1 (Retrieval)**: Bi-encoder similarity search backed by FAISS to retrieve the top candidate segments relevant to a user query.
3. **Phase 2 (Graph Construction & GAT Reasoning)**: Constructs an evidence graph where nodes are segments and edges are shared entities/semantic similarity. A trained Graph Attention Network traverses the graph to find multi-hop connections and weights the segments.
4. **Phase 3 (Answer Generation)**: Sends the top-weighted segments to a multi-provider LLM (Gemini/OpenAI/Groq) to generate a concise natural-language answer.
5. **XAI**: Computes SHAP values over the evidence segments, determining exactly which words were most influential in the model's retrieval.

## Setup & Installation

**Prerequisites**: Python 3.10, 3.11, or 3.12 is required (PyTorch 2.5 does not fully support Python 3.13+ yet).

1. Create and activate a Python virtual environment.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Create a `.env` file in the root directory and add your API keys:
   ```env
   PRIMARY_LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   ```

*Note on Windows*: If you encounter an `OSError: [WinError 1314]` when Whisper downloads its model, set these environment variables before running to bypass symlink restrictions:
```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
$env:HF_HUB_DISABLE_SYMLINKS="1"
```

## Running the Application

The project comes with a FastAPI backend and a test frontend UI to interact with it.

1. **Start the Backend**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

2. **Start the Frontend**:
   Open a new terminal and run:
   ```bash
   python -m http.server 3000 --directory test_frontend
   ```
   Navigate to [http://localhost:3000](http://localhost:3000) to use the web application.

### Using the Frontend
- **Upload Meeting**: Upload an audio or video file. The backend will process the SLP pipeline and return a Meeting ID.
- **Ask a Question**: Type a question about the meeting. The interface will display:
  1. The LLM's generated answer.
  2. An **Interactive Physics Graph**: A visual representation of the GAT evidence reasoning paths. You can drag nodes around; top nodes are colored red.
  3. **SHAP Evidence Highlights**: The top-ranked segments reconstructed word-by-word. Words with a high positive SHAP influence are highlighted in red, proving exactly *why* the AI chose them.

## Project Structure

```
Listen/
├── .env                        # API keys
├── config.py                   # Global hyperparameters
├── schemas.py                  # API schemas
├── app/
│   ├── main.py                 # FastAPI Application
│   ├── api/                    # API routing endpoints
│   ├── dl/                     # GAT Model, FAISS Retrieval, and Graph Builder
│   ├── services/               # Core pipeline orchestration
│   ├── slp/                    # Whisper ASR and spaCy processing
│   └── xai/                    # SHAP and Counterfactual explanation logic
├── data/                       # Stores uploaded media, transcripts, and indices
├── checkpoints/                # Saved model weights (hotpotqa_best.pt)
└── test_frontend/              # HTML/JS test application
```
