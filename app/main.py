"""FastAPI application entrypoint for the LISTEN backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import counterfactual, query, upload
from app.config import API_PREFIX
from app.models.schemas import HealthResponse

app = FastAPI(
    title="LISTEN Backend",
    description=(
        "Explainable multi-hop QA over real meeting recordings — "
        "Speech & Language Processing, Deep Learning (GAT), and XAI layers."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix=API_PREFIX)
app.include_router(query.router, prefix=API_PREFIX)
app.include_router(counterfactual.router, prefix=API_PREFIX)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe for deployment platforms."""
    return HealthResponse(status="ok", version=__version__)
