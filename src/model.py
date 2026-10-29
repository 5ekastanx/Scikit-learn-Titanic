"""Сборка модели: признаки -> подготовка -> случайный лес. Всё в одном Pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.features import add_features

# Колонки, которые должен дать пользователь (сырые данные)
RAW_COLS = ["Name", "Sex", "Age", "Pclass", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_RAW = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

# Колонки после add_features
NUM_COLS = ["Pclass", "Age", "Fare", "FamilySize"]
CAT_COLS = ["Sex", "Embarked", "Title", "IsAlone"]


def build_model(random_state: int = 42) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CAT_COLS),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),   # Title, FamilySize
        ("prep", preprocess),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                         random_state=random_state)),
    ])
