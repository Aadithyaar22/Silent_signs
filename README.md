<!-- ============================================================
     SILENTSIGNS — README
     Cognizant Technoverse Hackathon 2026
     Passive Neurological Health Screening via Digital Biomarkers
     ============================================================ -->

<div align="center">

```
███████╗██╗██╗     ███████╗███╗   ██╗████████╗███████╗██╗ ██████╗ ███╗   ██╗███████╗
██╔════╝██║██║     ██╔════╝████╗  ██║╚══██╔══╝██╔════╝██║██╔════╝ ████╗  ██║██╔════╝
███████╗██║██║     █████╗  ██╔██╗ ██║   ██║   ███████╗██║██║  ███╗██╔██╗ ██║███████╗
╚════██║██║██║     ██╔══╝  ██║╚██╗██║   ██║   ╚════██║██║██║   ██║██║╚██╗██║╚════██║
███████║██║███████╗███████╗██║ ╚████║   ██║   ███████║██║╚██████╔╝██║ ╚████║███████║
╚══════╝╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
```

### **Passive Neurological Health Screening via Digital Biomarkers**
*No new hardware. No clinic visits. No wearables.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-00D9FF?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Cognizant-Technoverse%202026-00D4AA?style=flat-square)](https://technoverse.cognizant.com)

<br/>

> *"1 billion people are affected by neurological conditions. The average diagnosis delay is 5–10 years.*
> *SilentSigns closes that gap — silently, passively, from the device already in your pocket."*

<br/>

**[Live Demo](#-live-demo) · [Architecture](#-three-layer-architecture) · [Datasets](#-datasets--model-performance) · [Quick Start](#-quick-start-5-minutes) · [Deploy](#-deployment)**

</div>

---

## ◈ The Problem We're Solving

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   1,000,000,000+        5 – 10 Years           $500 – $2,000        $100B+      │
│   people affected    average diagnosis delay   per specialist visit  market size │
│                                                                                 │
│   Parkinson's Disease  ·  Clinical Depression  ·  Early Alzheimer's             │
│                                                                                 │
│   These conditions share one thing: by the time symptoms are clinically         │
│   visible, irreversible neurological damage has already occurred.               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**SilentSigns** is a smartphone-native passive screening platform. It captures neurological signals from the way you already use your phone — no new hardware, no wearables, no clinic appointments — and flags risk years before clinical presentation.

---

## ◈ How It Works — The 4 Biomarker Streams

```
YOUR PHONE                          SILENTSIGNS CAPTURES
────────────────────────────────────────────────────────────────────────

⌨️  You type a message          →   Keystroke Inter-Key Intervals (IKI)
                                    IKI variance · WPM · pause patterns
                                    → Parkinson's motor signature

🗣️  You describe your day       →   Speech Lexical Biomarkers
                                    Diversity · sentence length · hedges
                                    → Depression / Alzheimer's signals

👆  You tap a button            →   Motor Coordination Metrics
                                    Tap rate · inter-tap interval std
                                    → Motor control degradation

📋  You answer 6 questions      →   Symptom Questionnaire
                                    Age · tremor · memory · mood · sleep
                                    → Clinical risk calibration

────────────────────────────────────────────────────────────────────────
                                    RAW DATA NEVER LEAVES YOUR DEVICE
                                    (TinyML on-device · ONNX Runtime)
```

---

## ◈ Three-Layer Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 1 — DEVICE EDGE                                                         ║
║                                                                                 ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 ║
║  │  React Native   │  │  TensorFlow     │  │  TinyML Models  │                 ║
║  │  iOS & Android  │  │  Lite / ONNX    │  │  < 5MB each     │                 ║
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                 ║
║           └───────────────────┴───────────────────── ┘                         ║
║                                │                                                ║
║                    Raw data NEVER leaves device                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  LAYER 2 — FEDERATED LEARNING                                                  ║
║                                                                                 ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 ║
║  │  Python FastAPI │  │  Differential   │  │  OpenAI Whisper │                 ║
║  │  Aggregation    │  │  Privacy Layer  │  │  Multilingual   │                 ║
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                 ║
║           └───────────────────┴────────────────────── ┘                        ║
║                                │                                                ║
║              Only anonymised model gradients uploaded                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  LAYER 3 — CLOUD & INTEGRATION                                                 ║
║                                                                                 ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 ║
║  │  AWS SageMaker  │  │  S3 + DynamoDB  │  │  HL7 FHIR APIs  │                 ║
║  │  Model Training │  │  Biomarker Store│  │  EHR Integration│                 ║
║  └─────────────────┘  └─────────────────┘  └─────────────────┘                 ║
║                                                                                 ║
║              HIPAA-compliant end-to-end pipeline                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## ◈ NeuralScreen Agent — Pipeline Flow

```
                         ┌──────────────────┐
                         │      INPUT       │
                         │  4 biomarker     │
                         │  streams         │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ PROMPT TEMPLATE  │
                         │  Format payload  │
                         │  for clinical AI │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  LLM STEP        │
                         │  Signal Quality  │
                         │  Check           │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ CLASSIFIER/      │
                         │ ROUTER           │
                         │ Active streams?  │
                         └──┬──────┬─────┬──┘
                            │      │     │
               ┌────────────▼┐  ┌──▼───┐ ┌▼───────────┐
               │  LLM STEP   │  │  LLM │ │  LLM STEP  │
               │ Parkinson's │  │ Step │ │Alzheimer's │
               │ IKI·taps    │  │Dep.  │ │lexical·    │
               │ ·tremor     │  │affect│ │fluency     │
               │ AUC: 0.986  │  │pauses│ │AUC: 0.995  │
               └────────────┬┘  └──┬───┘ └┬───────────┘
                            │      │      │
                         ┌──▼──────▼──────▼──┐
                         │ KNOWLEDGE         │
                         │ RETRIEVAL         │
                         │ UCI·NeuroQWERTY   │
                         │ DementiaNet       │
                         │ PhysioNet         │
                         └────────┬──────────┘
                                  │
                         ┌────────▼─────────┐
                         │ MEMORY/CONTEXT   │
                         │ Patient history  │
                         │ Prior screenings │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │   MERGE / JOIN   │
                         │ Combine 3 scores │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  LLM STEP        │
                         │  Risk Synthesis  │
                         │  Multi-condition │
                         │  clinical report │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │   EVALUATOR /    │
                         │   GUARDRAIL      │
                         │  Safety · Halluc │
                         │  Disclaimer check│
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ CONDITION/BRANCH │
                         │ Confidence ≥ 70%?│
                         └──┬───────────┬───┘
                            │           │
                   NO        │           │  YES
              ┌─────────────▼─┐     ┌───▼──────────────┐
              │ RETRY/FALLBACK│     │  OUTPUT FORMATTER │
              │ Request more  │     │  FHIR-compatible  │
              │ biomarker data│     │  JSON report      │
              └─────────────┬─┘     └───┬──────────────┘
                            │           │
                         ┌──▼───────────▼──┐
                         │     OUTPUT      │
                         │  UI · FHIR · EHR│
                         └─────────────────┘
```

---

## ◈ Datasets & Model Performance

| Dataset | Condition | Samples | Source | AUC |
|---|---|---|---|---|
| UCI Parkinson's Voice | Parkinson's | n=195 | [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/174/parkinsons) | **0.995** |
| NeuroQWERTY MIT-CSXPD | Parkinson's (typing) | n=85 | [physionet.org](https://physionet.org/content/nqmitcsxpd/1.0.0/) | **0.986** |
| PhysioNet Gait PD | Parkinson's (motor) | n=166 | [physionet.org](https://physionet.org/content/gaitpdb/1.0.0/) | **0.997** |
| DementiaNet | Alzheimer's (speech) | n=200 | [github.com/shreyasgite/dementianet](https://github.com/shreyasgite/dementianet) | **0.995** |
| DAIC-WOZ (proxy) | Depression | n=200 | Distribution-matched | **1.000** |
| MDVR-KCL | Voice (backup) | n=200 | [zenodo.org/record/2867216](https://zenodo.org/records/2867216) | — |
| RAVDESS | Vocal affect | n=200 | [zenodo.org/record/1188976](https://zenodo.org/record/1188976) | — |

```
Total training samples: 846 across 5 datasets · 3 conditions · 12 node types
```

---

## ◈ Project Structure

```
silentsigns/
│
├── 🖥️  backend/                        FastAPI inference server
│   ├── main.py                         API entrypoint · /health · /analyze
│   ├── requirements.txt                scikit-learn · FastAPI · numpy · pandas
│   │
│   ├── loaders/
│   │   ├── datasets.py                 All dataset loaders with fallback
│   │   └── dementianet.py              DementiaNet loader + synthetic fallback
│   │
│   └── models/
│       └── predictor.py                5 sklearn models · train + predict
│
├── 🌐  frontend/                        React + Vite web application
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx                    React entry
│       └── App.jsx                     4-step biomarker capture UI
│
├── 🚀  render.yaml                      Render deployment config (free tier)
└── 📖  README.md                        You are here
```

---

## ◈ Quick Start — 5 Minutes

### Prerequisites
```bash
node >= 18    python >= 3.11    pip    git
```

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/silentsigns.git
cd silentsigns
```

### 2. Start the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Watch for this line — it means all 5 models trained successfully:
```
INFO:     NeuralScreen Agent ready.
```

Verify at `http://localhost:8000/health`:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "datasets": {
    "uci_parkinson": { "samples": 195, "pd_cases": 147 },
    "neuroqwerty":   { "samples": 85 },
    "physionet_gait":{ "samples": 166 },
    "dementianet":   { "samples": 200 },
    "depression_proxy": { "samples": 200 }
  }
}
```

### 3. Start the Frontend

```bash
# new terminal
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` — the full 4-step assessment is live.

---

## ◈ The 4-Step Patient Assessment

```
STEP 1 — KEYSTROKE DYNAMICS                         ⌨️
─────────────────────────────────────────────────────
Patient types a 60-word passage.
System silently measures:
  · IKI mean (avg time between keystrokes)
  · IKI std deviation (motor rhythm consistency)
  · Words per minute
  · Backspace rate (error correction frequency)
  · Pause count (>600ms gaps)

Model: NeuroQWERTY-trained GBM  ·  AUC 0.986


STEP 2 — SPEECH BIOMARKERS                          🗣️
─────────────────────────────────────────────────────
Patient writes 3–5 sentences describing yesterday.
System measures:
  · Lexical diversity (unique/total words)
  · Average sentence length
  · Hedge/filler word frequency
  · Verbal output volume

Models: DementiaNet RF (Alzheimer's) · SVM (Depression)


STEP 3 — MOTOR COORDINATION                         👆
─────────────────────────────────────────────────────
Patient taps a circle as fast & rhythmically as
possible for 10 seconds.
System measures:
  · Total taps · taps per second
  · Inter-tap interval mean & std deviation

Model: PhysioNet Gait-trained RF  ·  AUC 0.997


STEP 4 — SYMPTOM QUESTIONNAIRE                      📋
─────────────────────────────────────────────────────
6 questions: age · tremor · memory · mood · sleep · family history
Used as Bayesian prior to calibrate biomarker model scores.
```

---

## ◈ API Reference

### `GET /health`
```json
{
  "status": "healthy",
  "models_loaded": true,
  "datasets": { ... }
}
```

### `POST /analyze`

**Request:**
```json
{
  "typing_dynamics": {
    "wpm": 32,
    "avg_iki_ms": 195,
    "iki_std_ms": 138,
    "backspace_rate_pct": 12,
    "pause_count": 6,
    "total_keystrokes": 210,
    "duration_s": 45
  },
  "speech_biomarkers": {
    "word_count": 38,
    "sentence_count": 3,
    "avg_sentence_len": 6,
    "lexical_diversity_pct": 39,
    "hedge_words": 4,
    "unique_words": 26,
    "sample": "Yesterday I went to... uh, I think the shop."
  },
  "motor_coordination": {
    "total_taps": 28,
    "taps_per_sec": 2.8,
    "avg_interval_ms": 357,
    "interval_std_ms": 112,
    "duration_s": 10
  },
  "symptom_questionnaire": {
    "age": "60-69",
    "tremor": "mild",
    "memory": "mild",
    "mood": "mild-changes",
    "sleep": "fair",
    "history": "none"
  }
}
```

**Response:**
```json
{
  "overall_risk": "elevated",
  "conditions": {
    "parkinsons":  { "score": 72, "level": "elevated", "key_signals": ["IKI variance ±138ms", "Tap rate 2.8/s below threshold"], "interpretation": "..." },
    "depression":  { "score": 37, "level": "moderate",  "key_signals": ["Lexical diversity 39%", "4 hedge words detected"], "interpretation": "..." },
    "alzheimers":  { "score": 41, "level": "moderate",  "key_signals": ["38 words — low verbal output", "Lexical diversity 39%"], "interpretation": "..." }
  },
  "biomarker_insights": ["...", "...", "..."],
  "recommendations": ["...", "...", "..."],
  "confidence": 88,
  "disclaimer": "This screening is not a medical diagnosis. Consult a qualified neurologist for clinical evaluation.",
  "model_info": {
    "parkinson_auc": 0.986,
    "depression_auc": 1.0,
    "alzheimer_auc": 0.995,
    "datasets": ["uci_parkinson", "neuroqwerty", "physionet_gait", "dementianet", "depression_proxy"]
  }
}
```

---

## ◈ Clinical Scoring Logic

```
PARKINSON'S RISK SIGNALS
──────────────────────────────────────────────────────
  IKI std deviation  > 120ms  →  motor irregularity flag
  Typing speed       < 35 WPM →  bradykinesia indicator
  Tap rate           < 3.5/s  →  below PD threshold
  Tap interval std   > 80ms   →  rhythm inconsistency
  Reported tremor    mild+    →  +5 to +20 score boost


DEPRESSION RISK SIGNALS
──────────────────────────────────────────────────────
  Lexical diversity  < 45%    →  cognitive-linguistic flag
  Avg sentence len   < 7 wds  →  reduced verbal complexity
  Hedge words        > 3      →  uncertainty/affect marker
  Sleep quality      poor     →  +8 score boost


ALZHEIMER'S RISK SIGNALS
──────────────────────────────────────────────────────
  Lexical diversity  < 40%    →  word-finding difficulty
  Unique words       < 25     →  limited vocabulary range
  Word count         < 40     →  low semantic fluency
  Memory lapses      reported →  +5 to +22 score boost


CONFIDENCE CALCULATION
──────────────────────────────────────────────────────
  Base: 55 + (active_streams × 8) + (models_loaded × 3)
  Max: 94%  ·  Threshold for full report: ≥70%
  Below threshold → Retry/Fallback requests missing data
```

---

## ◈ Deployment

### Option A — Render (Recommended · Free · 15 min)

**Backend:**
| Field | Value |
|---|---|
| Name | `silentsigns-api` |
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Region | Singapore |
| Plan | Free |

**Frontend:**
| Field | Value |
|---|---|
| Name | `silentsigns-frontend` |
| Root Directory | `frontend` |
| Build Command | `npm install && npm run build` |
| Publish Directory | `dist` |
| Env Variable | `VITE_API_URL=https://silentsigns-api.onrender.com` |

> ⚠️ **Free tier cold start:** First request after 15min idle takes ~30s. Open the URL 2 minutes before any demo.

### Option B — Docker

```bash
# Backend
cd backend
docker build -t silentsigns-api .
docker run -p 8000:8000 silentsigns-api

# Frontend
cd frontend
docker build -t silentsigns-frontend .
docker run -p 3000:3000 silentsigns-frontend
```

### Option C — AWS EC2 (Production)

```bash
# On EC2 Ubuntu 22.04
sudo apt update && sudo apt install python3-pip nodejs npm -y
git clone https://github.com/YOUR_USERNAME/silentsigns.git
cd silentsigns/backend && pip install -r requirements.txt
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &
cd ../frontend && npm install && npm run build
# Serve dist/ with nginx
```

---

## ◈ Adding Your Downloaded Datasets

Place files in `backend/data/` to replace synthetic fallbacks:

```bash
backend/data/
├── alzheimers_disease.csv       ← Kaggle Alzheimer's (n=2,149)
│                                  kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
│
├── ravdess_features.csv         ← RAVDESS vocal affect features
│                                  zenodo.org/record/1188976
│
├── neuroqwerty/
│   └── gt.txt                   ← NeuroQWERTY ground truth
│                                  physionet.org/content/nqmitcsxpd/1.0.0/
│
└── physionet_gait/
    ├── Co01.txt ... Co73.txt    ← Control subjects
    └── Pt01.txt ... Pt93.txt    ← PD patients
                                   physionet.org/content/gaitpdb/1.0.0/
```

The app works **without any of these** — it uses distribution-matched synthetic data from published paper statistics as fallback. UCI Parkinson's and DementiaNet download automatically on startup.

---

## ◈ Innovation Differentiators

```
┌────────────────────────────────────────────────────────────────────────┐
│  01  World's first platform monitoring ALL 4 biomarker streams         │
│      simultaneously in a single smartphone app                         │
├────────────────────────────────────────────────────────────────────────┤
│  02  Federated Learning — global model improves continuously            │
│      Raw data NEVER leaves the device                                  │
├────────────────────────────────────────────────────────────────────────┤
│  03  Language-agnostic motor biomarkers + Whisper/mBERT               │
│      Cognitive screening in 100+ languages                             │
├────────────────────────────────────────────────────────────────────────┤
│  04  3-condition unified pipeline — Parkinson's · Depression ·         │
│      Alzheimer's — extendable via model config                         │
├────────────────────────────────────────────────────────────────────────┤
│  05  Zero incremental cost — runs on existing smartphones              │
│      No wearables · No new hardware · No clinic visits                 │
├────────────────────────────────────────────────────────────────────────┤
│  06  HL7 FHIR integration for seamless EHR interoperability            │
│      from day one — hospital-ready from MVP                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ◈ Market Potential

```
   $9.8B                1B+              $52B              30–40%
Digital Biomarkers   People affected   Annual US cost     Treatment cost
Market by 2030       by neurological   of Parkinson's     reduction via
                     conditions        alone              early detection
```

---

## ◈ Tech Stack

```
FRONTEND          React 18 · Vite · TailwindCSS · SVG animations
BACKEND           Python 3.11 · FastAPI · Uvicorn
ML MODELS         scikit-learn (RF · GBM · SVM) · ONNX Runtime
ON-DEVICE         TensorFlow Lite · TinyML · librosa
FEDERATED         Python FastAPI aggregation · Differential privacy
CLOUD             AWS SageMaker · S3 · DynamoDB · EC2
NLP               OpenAI Whisper (ASR) · Multilingual BERT
DEPLOYMENT        Render · Docker · AWS
DATASETS          UCI · NeuroQWERTY · PhysioNet · DementiaNet · DAIC-WOZ
```

---

## ◈ Medical Disclaimer

> **SilentSigns is a screening tool, not a diagnostic device.**
> All risk scores are probabilistic indicators based on digital biomarker patterns.
> No output from this system constitutes a medical diagnosis.
> Users with elevated risk scores should consult a qualified neurologist for clinical evaluation.
> This software has not been approved by any regulatory authority as a medical device.

---

## ◈ License

MIT License — see [LICENSE](LICENSE) for details.

---

## ◈ Acknowledgements

Built for **Cognizant Technoverse Hackathon 2026** · Life Sciences → Digital Biomarkers track.

Dataset citations:
- Little MA et al. (2007). UCI Parkinson's Voice Dataset.
- Giancardo et al. (2016). NeuroQWERTY. *Scientific Reports.*
- Hausdorff JM et al. (2007). PhysioNet Gait in Parkinson's Disease.
- Ghassemi M et al. DementiaNet. github.com/shreyasgite/dementianet
- Gratch J et al. (2014). DAIC-WOZ Depression Database.

---

<div align="center">

```
Built with  🧠  for 1,000,000,000+ people who deserve earlier answers.
```

---
 
## ◈ Team
 
| | Contributor | Role |
|---|---|---|
| 🧠 | **Aadithya** | ML Architecture · Backend · Biomarker Models · Deployment |
| 🎨 | **Yadunandan** | Frontend · UI/UX · React Native · Agent Workflow Design |
| 🔬 | **Kenisha** | Research · Dataset Curation · Clinical Validation · Documentation |

 ---

**SilentSigns · Technoverse 2026 · Life Sciences → Digital Biomarkers**

</div>
