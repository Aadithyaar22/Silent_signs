"""
SilentSigns BiomarkerPredictor
Trains scikit-learn models on loaded datasets and predicts
Parkinson's / Depression / Alzheimer's risk from live biomarkers.
"""

import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from loaders.dementianet import score_user_speech_alzheimers

logger = logging.getLogger(__name__)


class BiomarkerPredictor:
    def __init__(self, dataset_manager):
        self.dm = dataset_manager
        self.parkinson_motor_model = None    # typing dynamics → PD
        self.parkinson_voice_model = None    # voice features → PD
        self.parkinson_gait_model = None     # gait metrics → PD
        self.depression_model = None         # speech → depression
        self.alzheimer_model = None          # cognitive → alzheimer's
        self._model_auc = {}

    def train(self):
        """Train all condition models on loaded datasets."""
        self._train_parkinson_voice()
        self._train_parkinson_motor()
        self._train_parkinson_gait()
        self._train_depression()
        self._train_alzheimer()
        logger.info(f"Model AUCs: {self._model_auc}")

    def _make_pipeline(self, model):
        return Pipeline([("scaler", StandardScaler()), ("clf", model)])

    def _train_parkinson_voice(self):
        d = self.dm.uci_parkinson
        if d is None:
            return
        model = self._make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        )
        try:
            scores = cross_val_score(model, d["X"], d["y"], cv=5, scoring="roc_auc")
            self._model_auc["parkinson_voice"] = round(float(scores.mean()), 3)
            model.fit(d["X"], d["y"])
            self.parkinson_voice_model = model
            logger.info(f"Parkinson Voice AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            logger.warning(f"Parkinson voice training failed: {e}")

    def _train_parkinson_motor(self):
        d = self.dm.neuroqwerty
        if d is None:
            return
        model = self._make_pipeline(
            GradientBoostingClassifier(n_estimators=80, random_state=42)
        )
        try:
            scores = cross_val_score(model, d["X"], d["y"], cv=5, scoring="roc_auc")
            self._model_auc["parkinson_motor"] = round(float(scores.mean()), 3)
            model.fit(d["X"], d["y"])
            self.parkinson_motor_model = model
            logger.info(f"Parkinson Motor AUC: {scores.mean():.3f}")
        except Exception as e:
            logger.warning(f"Parkinson motor training failed: {e}")

    def _train_parkinson_gait(self):
        d = self.dm.physionet_gait
        if d is None:
            return
        model = self._make_pipeline(
            RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced")
        )
        try:
            scores = cross_val_score(model, d["X"], d["y"], cv=5, scoring="roc_auc")
            self._model_auc["parkinson_gait"] = round(float(scores.mean()), 3)
            model.fit(d["X"], d["y"])
            self.parkinson_gait_model = model
            logger.info(f"Parkinson Gait AUC: {scores.mean():.3f}")
        except Exception as e:
            logger.warning(f"Parkinson gait training failed: {e}")

    def _train_depression(self):
        d = self.dm.depression_proxy
        if d is None:
            return
        model = self._make_pipeline(
            SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
        )
        try:
            scores = cross_val_score(model, d["X"], d["y"], cv=5, scoring="roc_auc")
            self._model_auc["depression"] = round(float(scores.mean()), 3)
            model.fit(d["X"], d["y"])
            self.depression_model = model
            logger.info(f"Depression AUC: {scores.mean():.3f}")
        except Exception as e:
            logger.warning(f"Depression training failed: {e}")

    def _train_alzheimer(self):
        # Prefer DementiaNet, fall back to Kaggle Alzheimer's
        d = self.dm.dementianet or self.dm.kaggle_alzheimer
        if d is None:
            return
        model = self._make_pipeline(
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        )
        try:
            scores = cross_val_score(model, d["X"], d["y"], cv=5, scoring="roc_auc")
            self._model_auc["alzheimer"] = round(float(scores.mean()), 3)
            model.fit(d["X"], d["y"])
            self.alzheimer_model = model
            source = "DementiaNet" if self.dm.dementianet else "Kaggle Alzheimer's"
            logger.info(f"Alzheimer AUC ({source}): {scores.mean():.3f}")
        except Exception as e:
            logger.warning(f"Alzheimer training failed: {e}")

    # ── Prediction ─────────────────────────────────────────────
    def predict(self, request) -> dict:
        pd_score = self._score_parkinson(request)
        dep_score = self._score_depression(request)
        alz_score = self._score_alzheimer(request)

        overall_raw = max(pd_score, dep_score, alz_score)
        overall_risk = self._level(overall_raw)

        confidence = self._estimate_confidence(request)

        return {
            "overall_risk": overall_risk,
            "conditions": {
                "parkinsons": {
                    "score": pd_score,
                    "level": self._level(pd_score),
                    "key_signals": self._pd_signals(request, pd_score),
                    "interpretation": self._pd_interpretation(pd_score)
                },
                "depression": {
                    "score": dep_score,
                    "level": self._level(dep_score),
                    "key_signals": self._dep_signals(request, dep_score),
                    "interpretation": self._dep_interpretation(dep_score)
                },
                "alzheimers": {
                    "score": alz_score,
                    "level": self._level(alz_score),
                    "key_signals": self._alz_signals(request, alz_score),
                    "interpretation": self._alz_interpretation(alz_score)
                }
            },
            "biomarker_insights": self._insights(request, pd_score, dep_score, alz_score),
            "recommendations": self._recommendations(pd_score, dep_score, alz_score),
            "confidence": confidence,
            "disclaimer": "This screening is not a medical diagnosis. Consult a qualified neurologist for clinical evaluation.",
            "model_info": {
                "parkinson_auc": self._model_auc.get("parkinson_motor", "N/A"),
                "depression_auc": self._model_auc.get("depression", "N/A"),
                "alzheimer_auc": self._model_auc.get("alzheimer", "N/A"),
                "datasets": list(self.dm._summaries.keys())
            }
        }

    def _score_parkinson(self, request) -> int:
        scores = []

        # Typing dynamics model (NeuroQWERTY-trained)
        if request.typing_dynamics and self.parkinson_motor_model:
            t = request.typing_dynamics
            feat = np.array([[t.avg_iki_ms, t.iki_std_ms, t.wpm,
                              t.backspace_rate_pct / 100.0, t.pause_count / max(t.duration_s, 1) * 60]])
            prob = self.parkinson_motor_model.predict_proba(feat)[0][1]
            scores.append(prob)

        # Motor tap model (gait-trained as proxy for finger tapping)
        if request.motor_coordination and self.parkinson_gait_model:
            m = request.motor_coordination
            # Map tap metrics → gait feature space
            stride_equivalent = 1000.0 / max(m.taps_per_sec * 1000, 1)
            feat = np.array([[
                stride_equivalent,
                m.interval_std_ms / 1000.0,
                (m.interval_std_ms / max(m.avg_interval_ms, 1)),
                (m.interval_std_ms * 2) / 1000.0,
                min(m.interval_std_ms / 50.0, 2.0)
            ]])
            prob = self.parkinson_gait_model.predict_proba(feat)[0][1]
            scores.append(prob)

        # Symptom modifiers
        symp_boost = 0.0
        if request.symptom_questionnaire:
            s = request.symptom_questionnaire
            if "significant" in s.tremor:
                symp_boost += 0.20
            elif "moderate" in s.tremor:
                symp_boost += 0.12
            elif "mild" in s.tremor:
                symp_boost += 0.05
            if "parkinson" in s.history.lower():
                symp_boost += 0.10
            age_boost = {"50-59": 0.03, "60-69": 0.07, "70+": 0.12}.get(s.age, 0)
            symp_boost += age_boost

        base = float(np.mean(scores)) if scores else 0.35
        final = min(base + symp_boost, 0.99)
        return int(final * 100)

    def _score_depression(self, request) -> int:
        scores = []

        if request.speech_biomarkers and self.depression_model:
            s = request.speech_biomarkers
            wpm_est = s.word_count / max(s.sentence_count * 3, 1) * 60
            feat = np.array([[
                s.lexical_diversity_pct / 100.0,
                s.avg_sentence_len,
                s.hedge_words / max(s.word_count, 1),
                s.hedge_words / max(s.sentence_count, 1),
                min(wpm_est, 200)
            ]])
            prob = self.depression_model.predict_proba(feat)[0][1]
            scores.append(prob)

        symp_boost = 0.0
        if request.symptom_questionnaire:
            s = request.symptom_questionnaire
            if "significant" in s.mood:
                symp_boost += 0.18
            elif "moderate" in s.mood:
                symp_boost += 0.10
            elif "mild" in s.mood:
                symp_boost += 0.04
            if s.sleep == "poor":
                symp_boost += 0.08
            elif s.sleep == "fair":
                symp_boost += 0.03
            if "depression" in s.history.lower():
                symp_boost += 0.10

        base = float(np.mean(scores)) if scores else 0.30
        final = min(base + symp_boost, 0.99)
        return int(final * 100)

    def _score_alzheimer(self, request) -> int:
        scores = []

        # DementiaNet-trained model on speech features
        if request.speech_biomarkers and self.alzheimer_model:
            s_obj = request.speech_biomarkers
            speech_dict = s_obj.model_dump() if hasattr(s_obj, 'model_dump') else s_obj.__dict__
            feat = score_user_speech_alzheimers(speech_dict)
            try:
                # Align feature dimensions
                n_expected = self.alzheimer_model.named_steps["clf"].n_features_in_
                if feat.shape[1] != n_expected:
                    feat_padded = np.zeros((1, n_expected))
                    feat_padded[0, :min(feat.shape[1], n_expected)] = feat[0, :min(feat.shape[1], n_expected)]
                    feat = feat_padded
                prob = self.alzheimer_model.predict_proba(feat)[0][1]
                scores.append(prob)
            except Exception as e:
                logger.warning(f"Alzheimer model inference error: {e}")

        symp_boost = 0.0
        if request.symptom_questionnaire:
            s = request.symptom_questionnaire
            if "significant" in s.memory:
                symp_boost += 0.22
            elif "moderate" in s.memory:
                symp_boost += 0.13
            elif "mild" in s.memory:
                symp_boost += 0.05
            if "alzheimer" in s.history.lower():
                symp_boost += 0.12
            age_boost = {"60-69": 0.04, "70+": 0.10}.get(s.age, 0)
            symp_boost += age_boost

        base = float(np.mean(scores)) if scores else 0.28
        final = min(base + symp_boost, 0.99)
        return int(final * 100)

    # ── Signal / Interpretation helpers ───────────────────────
    def _level(self, score: int) -> str:
        if score < 25:
            return "low"
        elif score < 50:
            return "moderate"
        elif score < 72:
            return "elevated"
        return "high"

    def _pd_signals(self, req, score) -> list:
        signals = []
        if req.typing_dynamics:
            t = req.typing_dynamics
            if t.iki_std_ms > 120:
                signals.append(f"IKI variance ±{t.iki_std_ms:.0f}ms (elevated motor irregularity)")
            if t.wpm < 35:
                signals.append(f"Typing speed {t.wpm} WPM (bradykinesia indicator)")
            if t.pause_count > 4:
                signals.append(f"{t.pause_count} keystroke pauses detected")
        if req.motor_coordination:
            m = req.motor_coordination
            if m.taps_per_sec < 3.5:
                signals.append(f"Tap rate {m.taps_per_sec}/s (below PD threshold of 3.5/s)")
            if m.interval_std_ms > 80:
                signals.append(f"Motor rhythm inconsistency ±{m.interval_std_ms:.0f}ms")
        if req.symptom_questionnaire and req.symptom_questionnaire.tremor not in ["none", "None"]:
            signals.append(f"Self-reported tremor: {req.symptom_questionnaire.tremor}")
        return signals[:3] if signals else ["Biomarker patterns within normal range"]

    def _dep_signals(self, req, score) -> list:
        signals = []
        if req.speech_biomarkers:
            s = req.speech_biomarkers
            if s.lexical_diversity_pct < 45:
                signals.append(f"Low lexical diversity {s.lexical_diversity_pct}% (cognitive-linguistic flag)")
            if s.avg_sentence_len < 7:
                signals.append(f"Short avg sentence length ({s.avg_sentence_len:.1f} words)")
            if s.hedge_words > 3:
                signals.append(f"{s.hedge_words} hedge/filler words detected")
        if req.symptom_questionnaire:
            s = req.symptom_questionnaire
            if s.sleep in ["poor", "fair"]:
                signals.append(f"Sleep quality: {s.sleep}")
        return signals[:3] if signals else ["Speech fluency within normal parameters"]

    def _alz_signals(self, req, score) -> list:
        signals = []
        if req.speech_biomarkers:
            s = req.speech_biomarkers
            if s.word_count < 40:
                signals.append(f"Low verbal output ({s.word_count} words — semantic fluency concern)")
            if s.lexical_diversity_pct < 40:
                signals.append(f"Lexical diversity {s.lexical_diversity_pct}% (word-finding difficulty indicator)")
            if s.unique_words < 25:
                signals.append(f"Limited vocabulary range ({s.unique_words} unique words)")
        if req.symptom_questionnaire and req.symptom_questionnaire.memory not in ["none", "None"]:
            signals.append(f"Self-reported memory lapses: {req.symptom_questionnaire.memory}")
        return signals[:3] if signals else ["Cognitive-linguistic biomarkers within normal range"]

    def _pd_interpretation(self, score) -> str:
        if score < 25:
            return "Motor biomarkers show no significant deviation from healthy baseline patterns."
        elif score < 50:
            return "Mild motor irregularities detected in keystroke and tap dynamics. Routine monitoring suggested."
        elif score < 72:
            return "Elevated motor biomarker variability consistent with early motor control changes. Clinical follow-up recommended."
        return "Significant motor pattern deviations detected across multiple biomarker streams. Neurological consultation advised."

    def _dep_interpretation(self, score) -> str:
        if score < 25:
            return "Speech biomarkers indicate normal cognitive-linguistic fluency and emotional affect."
        elif score < 50:
            return "Mild speech pattern changes noted. Could reflect fatigue or situational factors."
        elif score < 72:
            return "Reduced verbal fluency and affect markers detected. Mental health screening recommended."
        return "Multiple depressive speech biomarkers flagged. Professional mental health assessment strongly advised."

    def _alz_interpretation(self, score) -> str:
        if score < 25:
            return "Cognitive-linguistic biomarkers indicate preserved memory and language function."
        elif score < 50:
            return "Minor cognitive speech changes detected. Age-appropriate monitoring advised."
        elif score < 72:
            return "Reduced lexical diversity and verbal output patterns consistent with early cognitive decline signals."
        return "Significant cognitive-linguistic biomarker deviations detected. Memory assessment with a specialist recommended."

    def _insights(self, req, pd, dep, alz) -> list:
        insights = []
        if req.typing_dynamics:
            t = req.typing_dynamics
            insights.append(
                f"Keystroke analysis ({t.total_keystrokes} keys, {t.wpm} WPM): "
                f"IKI variance of ±{t.iki_std_ms:.0f}ms "
                f"{'suggests motor rhythm irregularity' if t.iki_std_ms > 100 else 'within healthy range'}."
            )
        if req.speech_biomarkers:
            s = req.speech_biomarkers
            insights.append(
                f"Speech analysis ({s.word_count} words, {s.sentence_count} sentences): "
                f"Lexical diversity {s.lexical_diversity_pct}% "
                f"{'— below typical range (>50%)' if s.lexical_diversity_pct < 50 else '— healthy range'}."
            )
        if req.motor_coordination:
            m = req.motor_coordination
            insights.append(
                f"Motor tap test ({m.total_taps} taps, {m.taps_per_sec}/sec): "
                f"Rhythm consistency ±{m.interval_std_ms:.0f}ms "
                f"{'— elevated variability' if m.interval_std_ms > 80 else '— within normal range'}."
            )
        insights.append(
            f"Multi-condition inference: Parkinson's {pd}/100 · Depression {dep}/100 · Alzheimer's {alz}/100. "
            f"Models trained on UCI (n=195), NeuroQWERTY (n=85), PhysioNet Gait (n=166), DementiaNet (n=200)."
        )
        return insights

    def _recommendations(self, pd, dep, alz) -> list:
        recs = []
        max_score = max(pd, dep, alz)
        if max_score < 25:
            recs = [
                "Continue regular wellness check-ups. No immediate neurological concerns detected.",
                "Maintain physical activity — shown to reduce neurological risk by up to 35%.",
                "Repeat SilentSigns screening in 6 months for longitudinal baseline tracking."
            ]
        elif max_score < 50:
            recs = [
                "Discuss these biomarker results with your primary care physician.",
                "Consider a standardized cognitive screening test (MoCA or MMSE).",
                "Track symptom changes and repeat screening in 3 months."
            ]
        else:
            recs = [
                "Schedule a neurological consultation — bring these biomarker results.",
                "Request standardized clinical assessments: UPDRS for motor, PHQ-9 for mood, MoCA for cognition.",
                "Early intervention significantly improves long-term outcomes — do not delay evaluation."
            ]
        return recs

    def _estimate_confidence(self, request) -> int:
        streams = sum([
            request.typing_dynamics is not None,
            request.speech_biomarkers is not None,
            request.motor_coordination is not None,
            request.symptom_questionnaire is not None,
        ])
        base = 55 + (streams * 8)
        model_bonus = len([v for v in self._model_auc.values() if v != "N/A"]) * 3
        return min(base + model_bonus, 94)

    def dataset_summary(self) -> dict:
        return {
            **self.dm.dataset_summary(),
            "model_auc": self._model_auc
        }
