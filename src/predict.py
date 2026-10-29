"""Предсказание для одного пассажира.  Демо:  python -m src.predict"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model import NUMERIC_RAW, RAW_COLS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    # Важно: та же версия scikit-learn, что при обучении (см. models/metrics.json)
    return joblib.load(path)


def predict_passenger(model, passenger: dict) -> dict:
    """passenger: словарь с сырыми полями. Пропущенные поля = пропуски (NaN)."""
    df = pd.DataFrame([passenger]).reindex(columns=RAW_COLS)    # нужные колонки, в нужном порядке
    df = df.astype(object).where(df.notna(), np.nan)             # None -> NaN
    df[NUMERIC_RAW] = df[NUMERIC_RAW].apply(pd.to_numeric, errors="coerce")  # числа снова числами
    proba = float(model.predict_proba(df)[0, 1])
    return {"survived": int(proba >= 0.5), "probability": round(proba, 3)}


if __name__ == "__main__":
    model = load_model()
    examples = {
        "Мужчина, 3-й класс, 22 года": {
            "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22, "Pclass": 3,
            "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
        "Женщина, 1-й класс, 38 лет": {
            "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)", "Sex": "female",
            "Age": 38, "Pclass": 1, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"},
        "Мальчик, 3-й класс, возраст неизвестен": {
            "Name": "Panula, Master. Eino Viljami", "Sex": "male", "Age": None, "Pclass": 3,
            "SibSp": 4, "Parch": 1, "Fare": 39.69, "Embarked": "S"},
    }
    for title, p in examples.items():
        print(f"{title}: {predict_passenger(model, p)}")
