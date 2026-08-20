# SilentSigns — Deployment Guide
## Cognizant Technoverse 2026 · MVP Build

---

## 🔴 Live Demo
- **App:** https://silentsigns-frontend.onrender.com
- **API:** https://silentsigns-api.onrender.com ([health](https://silentsigns-api.onrender.com/health) · [docs](https://silentsigns-api.onrender.com/docs))

Hosted on Render's free tier — auto-redeploys on every push to `main`. First load after 15 min of inactivity takes ~30s while the backend cold-starts.

---

## Project Structure

```
silentsigns/
├── backend/
│   ├── main.py                    ← FastAPI server
│   ├── loaders/
│   │   ├── datasets.py            ← All dataset loaders
│   │   └── dementianet.py         ← DementiaNet specific loader
│   ├── models/
│   │   └── predictor.py           ← ML models (sklearn)
│   └── requirements.txt
├── frontend/
│   ├── src/App.jsx                ← React UI (calls real API)
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── render.yaml                    ← Render deployment config
```

---

## Step 1 — Test Locally First

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/health — should show `models_loaded: true`
Visit http://localhost:8000/docs — FastAPI Swagger UI

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173

---

## Step 2 — Add Your Downloaded Datasets (Optional but Recommended)

Place datasets in `backend/data/`:

```bash
backend/data/
├── alzheimers_disease.csv         ← Kaggle Alzheimer's dataset
├── ravdess_features.csv           ← RAVDESS (if pre-processed)
├── neuroqwerty/
│   └── gt.txt                     ← NeuroQWERTY ground truth file
└── physionet_gait/
    ├── Co01.txt                   ← PhysioNet control subjects
    ├── Pt01.txt                   ← PhysioNet PD subjects
    └── ...
```

The app works without these (uses distribution-matched synthetic data as fallback).
UCI Parkinson's and DementiaNet are downloaded automatically on startup.

---

## Step 3 — Deploy to Render (Free Tier)

`render.yaml` at the repo root already defines both services, so Render can deploy them together via a **Blueprint** — no manual field-filling needed.

### 3a. Push to GitHub
```bash
git add .
git commit -m "Your commit message"
git push origin main
```
(Repo: https://github.com/Aadithyaar22/Silent_signs)

### 3b. Deploy via Render Blueprint

1. Go to https://render.com → Sign up / Log in
2. Click **"New +"** → **"Blueprint"**
3. Select the `Aadithyaar22/Silent_signs` repo
4. Render detects `render.yaml` and shows both services (`silentsigns-api`, `silentsigns-frontend`) — click **"Apply"**
5. Wait ~3-4 minutes for the backend build (installs deps + trains models) and ~2 minutes for the frontend build

Auto-deploy is on by default — every push to `main` triggers a fresh redeploy of both services.

### 3c. Your Live URLs
- Frontend: `https://silentsigns-frontend.onrender.com`
- Backend API: `https://silentsigns-api.onrender.com`
- API Docs: `https://silentsigns-api.onrender.com/docs`

---

## Step 4 — Verify Deployment

```bash
# Check backend health
curl https://silentsigns-api.onrender.com/health

# Expected response:
# {"status":"healthy","models_loaded":true,"datasets":{...}}

# Check dataset info
curl https://silentsigns-api.onrender.com/dataset-info
```

---

## ⚠️ Render Free Tier Notes

- **Cold start:** First request after 15min inactivity takes ~30 seconds (models reload)
- **Memory:** 512MB RAM — sklearn models fit comfortably
- **Tip for demo:** Open the app URL 2 minutes before presenting to judges so models are warm

---

## DementiaNet Integration Notes

DementiaNet (github.com/shreyasgite/dementianet) is loaded automatically:
1. App tries to download from GitHub on startup
2. If unavailable, uses distribution-matched synthetic features from the published paper
3. Either way, the Alzheimer's classifier trains and runs

To use real DementiaNet data:
```bash
git clone https://github.com/shreyasgite/dementianet
cp dementianet/data/* backend/data/
```

---

## Dataset AUC Reference (from literature)

| Model | Dataset | Expected AUC |
|---|---|---|
| Parkinson's Voice | UCI (n=195) | 0.86 |
| Parkinson's Motor | NeuroQWERTY (n=85) | 0.79-0.85 |
| Parkinson's Gait | PhysioNet (n=166) | 0.86 |
| Alzheimer's Speech | DementiaNet (n=200) | 0.72+ |
| Depression | DAIC-WOZ distribution | 0.76+ |

---

## Tech Stack (Cognizant Alignment)

| Category | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | Python FastAPI |
| ML | scikit-learn (RF, GBM, SVM) |
| Deployment | Render (AWS-compatible) |
| Data | UCI, PhysioNet, NeuroQWERTY, DementiaNet |
