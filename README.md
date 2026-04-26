# SilentSigns — Deployment Guide
## Cognizant Technoverse 2026 · MVP Build

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

### 3a. Push to GitHub
```bash
git init
git add .
git commit -m "SilentSigns MVP - Technoverse 2026"
git remote add origin https://github.com/YOUR_USERNAME/silentsigns.git
git push -u origin main
```

### 3b. Deploy Backend on Render

1. Go to https://render.com → Sign up / Log in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Fill in:
   - **Name:** `silentsigns-api`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
   - **Region:** Singapore (closest to Bangalore)
5. Click **"Create Web Service"**
6. Wait ~3-4 minutes for build + model training
7. Note your URL: `https://silentsigns-api.onrender.com`

### 3c. Deploy Frontend on Render

1. Click **"New +"** → **"Static Site"**
2. Connect same GitHub repo
3. Fill in:
   - **Name:** `silentsigns-frontend`
   - **Root Directory:** `frontend`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `dist`
4. Add environment variable:
   - **Key:** `VITE_API_URL`
   - **Value:** `https://silentsigns-api.onrender.com`
5. Click **"Create Static Site"**
6. Wait ~2 minutes

### 3d. Your Live URLs
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
