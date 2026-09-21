# LISTEN React frontend

This is the React/Vite interface for the LISTEN backend. It uses the existing API without requiring backend changes.

## Run locally

Start the FastAPI backend from the repository root first:

```powershell
uvicorn app.main:app --reload --port 8000
```

Then, in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite (normally `http://localhost:5173`).

The API base defaults to `http://127.0.0.1:8000/api/v1`. To point the frontend at a deployed API, make a `frontend/.env` file:

```env
VITE_API_BASE=https://your-api.example.com/api/v1
```
