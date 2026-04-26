from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import os
import logging

from loaders.datasets import DatasetManager
from models.predictor import BiomarkerPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SilentSigns NeuralScreen API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global predictor (loaded on startup) ──────────────────────
predictor: BiomarkerPredictor = None

@app.on_event("startup")
async def startup():
    global predictor
    logger.info("Loading datasets and training models...")
    dm = DatasetManager()
    dm.load_all()
    predictor = BiomarkerPredictor(dm)
    predictor.train()
    logger.info("NeuralScreen Agent ready.")


# ── Request / Response schemas ────────────────────────────────
class TypingMetrics(BaseModel):
    wpm: float
    avg_iki_ms: float
    iki_std_ms: float
    backspace_rate_pct: float
    pause_count: int
    total_keystrokes: int
    duration_s: float

class SpeechMetrics(BaseModel):
    word_count: int
    sentence_count: int
    avg_sentence_len: float
    lexical_diversity_pct: float
    hedge_words: int
    unique_words: int
    sample: str

class MotorMetrics(BaseModel):
    total_taps: int
    taps_per_sec: float
    avg_interval_ms: float
    interval_std_ms: float
    duration_s: float

class SymptomProfile(BaseModel):
    age: str
    tremor: str
    memory: str
    mood: str
    sleep: str
    history: str

class BiomarkerRequest(BaseModel):
    typing_dynamics: Optional[TypingMetrics] = None
    speech_biomarkers: Optional[SpeechMetrics] = None
    motor_coordination: Optional[MotorMetrics] = None
    symptom_questionnaire: Optional[SymptomProfile] = None

class ConditionScore(BaseModel):
    score: int
    level: str
    key_signals: list[str]
    interpretation: str

class RiskReport(BaseModel):
    overall_risk: str
    conditions: dict
    biomarker_insights: list[str]
    recommendations: list[str]
    confidence: int
    disclaimer: str
    model_info: dict


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "SilentSigns NeuralScreen API running", "version": "2.1.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "datasets": predictor.dataset_summary() if predictor else {}
    }

@app.post("/analyze", response_model=RiskReport)
def analyze(request: BiomarkerRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    try:
        report = predictor.predict(request)
        return report
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset-info")
def dataset_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return predictor.dataset_summary()
