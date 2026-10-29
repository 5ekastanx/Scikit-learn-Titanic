"""Обучение модели.  Запуск из корня проекта:  python -m src.train"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.model import RAW_COLS, build_model

ROOT = Path(__file__).resolve().parents[1]


def main(data_path: Path, model_path: Path, metrics_path: Path) -> dict:
    df = pd.read_csv(data_path)
    X, y = df[RAW_COLS], df["Survived"]

    # 1. Откладываем тест. До самого конца на него не смотрим.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Честная оценка: кросс-валидация только на учебной части.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r = cross_validate(build_model(), X_train, y_train, cv=cv,
                       scoring=["accuracy", "roc_auc"])

    # 3. Один раз проверяем на тесте.
    model = build_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "cv_accuracy": round(r["test_accuracy"].mean(), 3),
        "cv_accuracy_std": round(r["test_accuracy"].std(), 3),
        "cv_auc": round(r["test_roc_auc"].mean(), 3),
        "test_accuracy": round(accuracy_score(y_test, proba >= 0.5), 3),
        "test_auc": round(roc_auc_score(y_test, proba), 3),
        "baseline_accuracy": round(1 - y_test.mean(), 3),   # «все погибли»
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sklearn_version": sklearn.__version__,
    }

    # 4. Для боевой модели учим заново на всех данных (больше данных = лучше).
    final_model = build_model().fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "titanic.csv")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "model.joblib")
    p.add_argument("--metrics", type=Path, default=ROOT / "models" / "metrics.json")
    args = p.parse_args()
    for k, v in main(args.data, args.model, args.metrics).items():
        print(f"{k:18s} {v}")
