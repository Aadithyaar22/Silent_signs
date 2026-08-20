from fastapi.testclient import TestClient

from main import app


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert body["models_loaded"] is True


def test_analyze_with_symptom_questionnaire_only():
    with TestClient(app) as client:
        payload = {
            "symptom_questionnaire": {
                "age": "60-69",
                "tremor": "mild",
                "memory": "none",
                "mood": "none",
                "sleep": "good",
                "history": "none",
            }
        }
        res = client.post("/analyze", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["overall_risk"] in {"low", "moderate", "elevated", "high"}
        assert set(body["conditions"]) == {"parkinsons", "depression", "alzheimers"}


def test_analyze_requires_at_least_valid_body():
    with TestClient(app) as client:
        res = client.post("/analyze", json={})
        assert res.status_code == 200


def test_dataset_info():
    with TestClient(app) as client:
        res = client.get("/dataset-info")
        assert res.status_code == 200
        assert isinstance(res.json(), dict)
