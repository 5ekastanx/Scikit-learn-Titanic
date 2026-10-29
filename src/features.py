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
    title = (df["Name"].str.split(", ").str[1]       # «Mr. Owen Harris»
                       .str.split(" ").str[0]        # «Mr.»
                       .str.rstrip("."))             # «Mr»
    title = title.replace(TITLE_FIX)
    # редкие и пропущенные титулы -> «Rare»
    df["Title"] = title.where(title.isin(KEEP_TITLES), "Rare")

    # братья/сёстры/супруги + родители/дети + сам пассажир
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df
