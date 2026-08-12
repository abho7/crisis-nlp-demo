"""FastAPI service for the crisis-text classification demo.

Serves a single-page frontend plus POST /predict. The model behind it is a
stand-in trained on public data -- see README.md and model_adapter.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model_adapter import LinearStandInPredictor, Predictor

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_INPUT_CHARS = 2000

# A short, non-negotiable banner echoed on every response. The frontend renders
# it; keeping it server-side means an embedder cannot quietly drop it.
STANDIN_NOTICE = (
    "Stand-in model trained on the public Kaggle 'Real or Not? NLP with Disaster "
    "Tweets' dataset. This is NOT the model from the NHSJS 2025 paper, and its "
    "accuracy is not comparable to the results reported there. It predicts "
    "whether text is disaster-related, not how urgent it is."
)

state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup so the first request is not penalised, and so a
    # missing artifact fails loudly at boot instead of mid-demo.
    state["predictor"] = LinearStandInPredictor()
    yield
    state.clear()


app = FastAPI(
    title="Crisis Text Classification Demo",
    description=STANDIN_NOTICE,
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw message text to classify.")


class SegmentOut(BaseModel):
    text: str
    contribution: float
    is_token: bool


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    segments: list[SegmentOut]
    base_value: float
    absent_contribution: float
    logit: float
    model_kind: str
    is_standin: bool
    notice: str


@app.get("/health")
def health() -> dict:
    predictor = state.get("predictor")
    return {
        "status": "ok" if predictor else "model not loaded",
        "model_kind": getattr(predictor, "_pipeline", None) and "tfidf+logreg (stand-in)",
        "classes": predictor.class_names if predictor else [],
    }


@app.get("/metrics")
def metrics() -> dict:
    """Held-out test metrics for the stand-in, as measured at training time."""
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"notice": STANDIN_NOTICE, **getattr(predictor, "metrics", {})}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Text must be {MAX_INPUT_CHARS} characters or fewer (got {len(text)}).",
        )

    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    result = predictor.predict(text)
    payload = asdict(result)
    payload["notice"] = STANDIN_NOTICE
    return PredictResponse(**payload)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
