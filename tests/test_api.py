from fastapi.testclient import TestClient

from app.main import app

OK = {"Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22,
      "Pclass": 3, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}


def test_health():
    with TestClient(app) as client:        # with = запускает lifespan (грузит модель)
        assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok():
    with TestClient(app) as client:
        r = client.post("/predict", json=OK)
        assert r.status_code == 200
        assert 0 <= r.json()["probability"] <= 1


def test_predict_validation_error():
    with TestClient(app) as client:
        r = client.post("/predict", json={**OK, "Sex": "robot"})
        assert r.status_code == 422