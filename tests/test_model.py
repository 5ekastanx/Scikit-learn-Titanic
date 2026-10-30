"""Быстрые тесты. Запуск из корня проекта:  pytest"""
import pandas as pd
from src.features import add_features
from src.model import RAW_COLS, build_model
from src.predict import predict_passenger


def tiny_data() -> pd.DataFrame:
    """Маленькая выдуманная таблица (настоящие данные в репозиторий не кладём)."""
    rows = []
    for i in range(10):
        rows.append(dict(Name=f"Smith{i}, Mr. John", Sex="male", Age=20 + i, Pclass=3,
                         SibSp=0, Parch=0, Fare=7.5 + i, Embarked="S", Survived=0))
        rows.append(dict(Name=f"Brown{i}, Mrs. Anna", Sex="female", Age=25 + i, Pclass=1,
                         SibSp=1, Parch=0, Fare=70.0 + i, Embarked="C", Survived=1))
    return pd.DataFrame(rows)


def test_add_features_title_and_family():
    df = pd.DataFrame([
        dict(Name="Braund, Mr. Owen Harris", SibSp=1, Parch=0),
        dict(Name="Heikkinen, Miss. Laina", SibSp=0, Parch=0),
        dict(Name="Dr. Without Comma", SibSp=2, Parch=3),
    ])
    out = add_features(df)
    assert list(out["Title"]) == ["Mr", "Miss", "Rare"]
    assert list(out["FamilySize"]) == [2, 1, 6]
    assert "Title" not in df.columns          # исходная таблица не изменилась


def test_add_features_name_without_title_separator():
    df = pd.DataFrame([dict(Name="No comma in this name", SibSp=0, Parch=0)])

    assert list(add_features(df)["Title"]) == ["Rare"]


def test_predict_returns_probability():
    data = tiny_data()
    model = build_model().fit(data[RAW_COLS], data["Survived"])
    out = predict_passenger(model, {"Name": "Doe, Mrs. Jane", "Sex": "female", "Age": 30,
                                    "Pclass": 1, "SibSp": 0, "Parch": 0, "Fare": 80,
                                    "Embarked": "C"})
    assert set(out) == {"survived", "probability"}
    assert 0.0 <= out["probability"] <= 1.0
    assert out["survived"] in (0, 1)


def test_predict_handles_missing_and_unknown():
    data = tiny_data()
    model = build_model().fit(data[RAW_COLS], data["Survived"])
    out = predict_passenger(model, {"Name": "Doe, Mr. John", "Sex": "male", "Age": None,
                                    "Pclass": 3, "SibSp": 0, "Parch": 0, "Fare": None,
                                    "Embarked": "X"})           # порт X модель не видела
    assert 0.0 <= out["probability"] <= 1.0

def test_is_alone():
    df = pd.DataFrame([
        dict(Name="Doe, Mr. John", SibSp=0, Parch=0),
        dict(Name="Doe, Mrs. Jane", SibSp=1, Parch=0),
    ])  
    out = add_features(df)
    assert list(out["IsAlone"]) == [1, 0]