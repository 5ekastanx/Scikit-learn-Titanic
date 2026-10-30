"""Признаки из «сырых» колонок Титаника.

Одна и та же функция работает и при обучении, и при предсказании.
Так мы не получим расхождение между обучением и боевым режимом.
"""
import pandas as pd

KEEP_TITLES = ["Mr", "Mrs", "Miss", "Master"]
TITLE_FIX = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонки Title и FamilySize. Исходную таблицу не меняет."""
    df = df.copy()                                   # не трогаем таблицу вызывающего

    # «Braund, Mr. Owen Harris» -> «Mr»
    title = (df["Name"].astype("string")
                       .str.extract(r",\s*([^\s]+)", expand=False)
                       .astype("string").str.rstrip("."))
    title = title.replace(TITLE_FIX)
    df["Title"] = title.where(title.isin(KEEP_TITLES), "Rare")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df
