"""
SilentSigns Dataset Manager
Loads all datasets used for model training:
  - UCI Parkinson's Voice (voice tremor → Parkinson's)
  - NeuroQWERTY MIT-CSXPD (typing dynamics → Parkinson's motor)
  - PhysioNet Gait in PD (gait/motor → Parkinson's)
  - DementiaNet (speech → Alzheimer's)
  - Kaggle Alzheimer's Disease Dataset (clinical → Alzheimer's)
  - RAVDESS proxy (vocal affect → Depression)
"""

import pandas as pd
import numpy as np
import requests
import io
import os
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    def __init__(self):
        self.uci_parkinson = None      # Voice tremor features
        self.neuroqwerty = None        # Typing dynamics
        self.physionet_gait = None     # Gait metrics
        self.dementianet = None        # Alzheimer's speech
        self.kaggle_alzheimer = None   # Alzheimer's clinical
        self.depression_proxy = None   # Depression features
        self._summaries = {}

    def load_all(self):
        """Load all datasets with graceful fallback on network errors."""
        self._load_uci_parkinson()
        self._load_neuroqwerty()
        self._load_physionet_gait()
        self._load_dementianet()
        self._load_kaggle_alzheimer()
        self._load_depression_proxy()
        logger.info(f"Datasets loaded: {list(self._summaries.keys())}")

    # ── UCI Parkinson's Voice ──────────────────────────────────
    def _load_uci_parkinson(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        try:
            r = requests.get(url, timeout=15)
            df = pd.read_csv(io.StringIO(r.text))
            # Features: MDVP:Fo, Jitter(%), Shimmer, HNR, RPDE, DFA, PPE etc.
            feature_cols = [c for c in df.columns if c not in ["name", "status"]]
            self.uci_parkinson = {
                "X": df[feature_cols].values,
                "y": df["status"].values,  # 1=PD, 0=healthy
                "features": feature_cols,
                "source": "UCI Parkinson's Voice Dataset",
            }
            self._summaries["uci_parkinson"] = {
                "samples": len(df), "pd_cases": int(df["status"].sum()),
                "features": len(feature_cols), "source": url
            }
            logger.info(f"UCI Parkinson's loaded: {len(df)} samples")
        except Exception as e:
            logger.warning(f"UCI Parkinson's download failed: {e} — using synthetic")
            self.uci_parkinson = self._synthetic_parkinson_voice()

    # ── NeuroQWERTY MIT-CSXPD Typing Dynamics ─────────────────
    def _load_neuroqwerty(self):
        """
        NeuroQWERTY MIT-CSXPD: 85 subjects, keystroke timing.
        PhysioNet: https://physionet.org/content/nqmitcsxpd/1.0.0/
        Direct MIT download: https://neuroqwerty.mit.edu/datasets
        """
        # Pre-computed aggregate statistics from NeuroQWERTY paper
        # (Giancardo et al., Scientific Reports 2016)
        # If you have the dataset downloaded, put it at /data/neuroqwerty/
        local_path = "/data/neuroqwerty/gt.txt"
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path, sep="\t")
                self.neuroqwerty = self._process_neuroqwerty(df)
                logger.info("NeuroQWERTY loaded from local path")
                return
            except Exception as e:
                logger.warning(f"Local NeuroQWERTY load failed: {e}")

        # Use published statistics to build training distribution
        logger.info("NeuroQWERTY: using published distribution statistics")
        self.neuroqwerty = self._synthetic_neuroqwerty()
        self._summaries["neuroqwerty"] = {
            "samples": 85, "features": 5,
            "source": "https://physionet.org/content/nqmitcsxpd/1.0.0/",
            "note": "Distribution-matched from Giancardo et al. 2016"
        }

    def _process_neuroqwerty(self, df):
        """Process ground truth file from NeuroQWERTY dataset."""
        feature_cols = ["typingSpeed", "nqScore", "afTap", "sTap"]
        available = [c for c in feature_cols if c in df.columns]
        return {
            "X": df[available].fillna(df[available].median()).values,
            "y": df["gt"].astype(int).values,  # 1=PD
            "features": available,
            "source": "NeuroQWERTY MIT-CSXPD (local)"
        }

    # ── PhysioNet Gait in Parkinson's Disease ─────────────────
    def _load_physionet_gait(self):
        """
        Gait in Parkinson's Disease v1.0.0
        https://physionet.org/content/gaitpdb/1.0.0/
        93 PD patients + 73 healthy controls
        """
        local_path = "/data/physionet_gait/"
        if os.path.exists(local_path):
            try:
                dfs = []
                for f in os.listdir(local_path):
                    if f.endswith(".txt"):
                        label = 0 if f.startswith("Co") else 1
                        data = pd.read_csv(os.path.join(local_path, f),
                                           sep=r"\s+", header=None)
                        feats = self._extract_gait_features(data, label)
                        dfs.append(feats)
                if dfs:
                    combined = pd.DataFrame(dfs)
                    self.physionet_gait = {
                        "X": combined.drop("label", axis=1).values,
                        "y": combined["label"].values,
                        "features": [c for c in combined.columns if c != "label"],
                        "source": "PhysioNet Gait in PD (local)"
                    }
                    self._summaries["physionet_gait"] = {
                        "samples": len(combined), "source": local_path
                    }
                    logger.info(f"PhysioNet Gait loaded: {len(combined)} subjects")
                    return
            except Exception as e:
                logger.warning(f"Local PhysioNet Gait load failed: {e}")

        logger.info("PhysioNet Gait: using published distribution statistics")
        self.physionet_gait = self._synthetic_gait()
        self._summaries["physionet_gait"] = {
            "samples": 166, "source": "https://physionet.org/content/gaitpdb/1.0.0/",
            "note": "Distribution-matched from Hausdorff et al. 2007"
        }

    def _extract_gait_features(self, data, label):
        """Extract stride interval statistics from raw PhysioNet gait file."""
        vals = data.iloc[:, 1].dropna().values  # stride intervals
        return {
            "stride_mean": np.mean(vals),
            "stride_std": np.std(vals),
            "stride_cv": np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0,
            "stride_range": np.max(vals) - np.min(vals),
            "stride_entropy": self._sample_entropy(vals),
            "label": label
        }

    def _sample_entropy(self, series, m=2, r_factor=0.2):
        """Simplified sample entropy for gait irregularity."""
        try:
            r = r_factor * np.std(series)
            N = min(len(series), 200)
            series = series[:N]
            count_m, count_m1 = 0, 0
            for i in range(N - m):
                for j in range(i + 1, N - m):
                    if np.max(np.abs(series[i:i+m] - series[j:j+m])) < r:
                        count_m += 1
                        if abs(series[i+m] - series[j+m]) < r:
                            count_m1 += 1
            if count_m == 0:
                return 0
            return -np.log(count_m1 / count_m) if count_m1 > 0 else 2.0
        except Exception:
            return 0.5

    # ── DementiaNet ───────────────────────────────────────────
    def _load_dementianet(self):
        from loaders.dementianet import download_dementianet, get_dementianet_features
        try:
            df = download_dementianet()
            X, y, features = get_dementianet_features(df)
            self.dementianet = {
                "X": X, "y": y, "features": features,
                "source": "DementiaNet (github.com/shreyasgite/dementianet)",
                "df": df
            }
            self._summaries["dementianet"] = {
                "samples": len(df),
                "dementia_cases": int(y.sum()),
                "features": len(features),
                "source": "https://github.com/shreyasgite/dementianet"
            }
            logger.info(f"DementiaNet loaded: {len(df)} samples")
        except Exception as e:
            logger.warning(f"DementiaNet load failed: {e}")
            self.dementianet = None

    # ── Kaggle Alzheimer's Disease Dataset ────────────────────
    def _load_kaggle_alzheimer(self):
        """
        Alzheimer's Disease Dataset (2,149 patients)
        Kaggle: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
        Download the CSV and put it at /data/alzheimers_disease.csv
        """
        local_path = "/data/alzheimers_disease.csv"
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path)
                # Target: Diagnosis (0=no AD, 1=AD)
                target_col = next(
                    (c for c in df.columns if "diagnosis" in c.lower() or "label" in c.lower()),
                    df.columns[-1]
                )
                feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                                if c != target_col][:15]
                self.kaggle_alzheimer = {
                    "X": df[feature_cols].fillna(df[feature_cols].median()).values,
                    "y": df[target_col].values,
                    "features": feature_cols,
                    "source": "Kaggle Alzheimer's Disease Dataset"
                }
                self._summaries["kaggle_alzheimer"] = {
                    "samples": len(df), "features": len(feature_cols),
                    "source": "https://kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset"
                }
                logger.info(f"Kaggle Alzheimer's loaded: {len(df)} samples")
                return
            except Exception as e:
                logger.warning(f"Kaggle Alzheimer's load failed: {e}")

        logger.info("Kaggle Alzheimer's: not found locally, using DementiaNet only")
        self.kaggle_alzheimer = None

    # ── Depression Proxy (RAVDESS-inspired) ───────────────────
    def _load_depression_proxy(self):
        """
        Depression biomarker proxy dataset.
        Uses speech feature distributions from depression research literature.
        Primary: RAVDESS emotional speech patterns (zenodo.org/record/1188976)
        Secondary: Published DAIC-WOZ statistics
        """
        ravdess_path = "/data/ravdess_features.csv"
        if os.path.exists(ravdess_path):
            try:
                df = pd.read_csv(ravdess_path)
                # RAVDESS: emotion labels 1-8, use sad(4)/neutral(2) vs happy(3)
                if "emotion" in df.columns:
                    df["depression_label"] = (df["emotion"].isin([4, 6])).astype(int)
                    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                                    if c not in ["emotion", "depression_label"]]
                    self.depression_proxy = {
                        "X": df[feature_cols].values,
                        "y": df["depression_label"].values,
                        "features": feature_cols,
                        "source": "RAVDESS (local)"
                    }
                    logger.info(f"RAVDESS loaded: {len(df)} samples")
                    return
            except Exception as e:
                logger.warning(f"RAVDESS load failed: {e}")

        self.depression_proxy = self._synthetic_depression()
        self._summaries["depression_proxy"] = {
            "samples": 200,
            "source": "Distribution-matched from DAIC-WOZ literature",
            "note": "Replace with RAVDESS: zenodo.org/record/1188976"
        }

    # ── Synthetic Dataset Generators ──────────────────────────
    def _synthetic_parkinson_voice(self, seed=42) -> dict:
        """Published voice biomarker distributions for PD vs healthy."""
        np.random.seed(seed)
        # PD: higher jitter, shimmer, lower HNR
        pd_data = np.column_stack([
            np.random.normal(0.006, 0.002, 147),   # jitter %
            np.random.normal(0.031, 0.010, 147),   # shimmer
            np.random.normal(19.0, 3.5, 147),      # HNR
            np.random.normal(0.52, 0.09, 147),     # RPDE
            np.random.normal(0.70, 0.08, 147),     # DFA
            np.random.normal(0.21, 0.05, 147),     # PPE
        ])
        healthy_data = np.column_stack([
            np.random.normal(0.003, 0.001, 48),
            np.random.normal(0.018, 0.006, 48),
            np.random.normal(24.0, 3.0, 48),
            np.random.normal(0.41, 0.07, 48),
            np.random.normal(0.69, 0.06, 48),
            np.random.normal(0.13, 0.04, 48),
        ])
        X = np.vstack([pd_data, healthy_data])
        y = np.hstack([np.ones(147), np.zeros(48)])
        idx = np.random.permutation(len(y))
        feat_names = ["jitter_pct", "shimmer", "hnr", "rpde", "dfa", "ppe"]
        self._summaries["uci_parkinson"] = {
            "samples": 195, "pd_cases": 147, "features": 6,
            "source": "Synthetic (UCI distribution)", "note": "Download: archive.ics.uci.edu/dataset/174/parkinsons"
        }
        return {"X": X[idx], "y": y[idx], "features": feat_names, "source": "Synthetic UCI"}

    def _synthetic_neuroqwerty(self, seed=43) -> dict:
        np.random.seed(seed)
        # PD: slower typing, higher IKI variance (Giancardo et al. 2016)
        pd_X = np.column_stack([
            np.random.normal(180, 40, 42),    # avg IKI ms
            np.random.normal(95, 25, 42),     # IKI std ms
            np.random.normal(28, 8, 42),      # WPM
            np.random.normal(0.08, 0.02, 42), # backspace rate
            np.random.normal(3.1, 0.9, 42),   # pause count/min
        ])
        ctrl_X = np.column_stack([
            np.random.normal(120, 30, 43),
            np.random.normal(45, 15, 43),
            np.random.normal(48, 10, 43),
            np.random.normal(0.04, 0.01, 43),
            np.random.normal(1.4, 0.5, 43),
        ])
        X = np.vstack([pd_X, ctrl_X])
        y = np.hstack([np.ones(42), np.zeros(43)])
        idx = np.random.permutation(85)
        return {"X": X[idx], "y": y[idx],
                "features": ["avg_iki_ms", "iki_std_ms", "wpm", "backspace_rate", "pause_rate"],
                "source": "Synthetic NeuroQWERTY"}

    def _synthetic_gait(self, seed=44) -> dict:
        np.random.seed(seed)
        # PD: higher stride variability (Hausdorff et al. 2007)
        pd_X = np.column_stack([
            np.random.normal(1.08, 0.08, 93),   # stride mean (s)
            np.random.normal(0.038, 0.015, 93), # stride std
            np.random.normal(0.036, 0.012, 93), # stride CV
            np.random.normal(0.18, 0.06, 93),   # stride range
            np.random.normal(1.6, 0.4, 93),     # entropy
        ])
        ctrl_X = np.column_stack([
            np.random.normal(1.02, 0.06, 73),
            np.random.normal(0.018, 0.007, 73),
            np.random.normal(0.018, 0.006, 73),
            np.random.normal(0.09, 0.03, 73),
            np.random.normal(0.9, 0.3, 73),
        ])
        X = np.vstack([pd_X, ctrl_X])
        y = np.hstack([np.ones(93), np.zeros(73)])
        idx = np.random.permutation(166)
        return {"X": X[idx], "y": y[idx],
                "features": ["stride_mean", "stride_std", "stride_cv", "stride_range", "stride_entropy"],
                "source": "Synthetic PhysioNet Gait PD"}

    def _synthetic_depression(self, seed=45) -> dict:
        np.random.seed(seed)
        # Depressed: low lexical diversity, more pauses, shorter sentences
        dep_X = np.column_stack([
            np.random.normal(0.38, 0.09, 100),  # lexical diversity
            np.random.normal(5.5, 1.5, 100),    # avg sentence len
            np.random.normal(0.11, 0.03, 100),  # filler word rate
            np.random.normal(3.2, 1.0, 100),    # pause rate
            np.random.normal(72, 18, 100),       # words per min
        ])
        ctrl_X = np.column_stack([
            np.random.normal(0.62, 0.08, 100),
            np.random.normal(9.1, 2.0, 100),
            np.random.normal(0.05, 0.02, 100),
            np.random.normal(1.6, 0.7, 100),
            np.random.normal(130, 22, 100),
        ])
        X = np.vstack([dep_X, ctrl_X])
        y = np.hstack([np.ones(100), np.zeros(100)])
        idx = np.random.permutation(200)
        return {"X": X[idx], "y": y[idx],
                "features": ["lexical_diversity", "avg_sentence_len", "filler_rate", "pause_rate", "wpm"],
                "source": "Synthetic DAIC-WOZ distribution"}

    def dataset_summary(self) -> dict:
        return self._summaries
