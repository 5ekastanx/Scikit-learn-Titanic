from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_model, predict_passenger


class Passenger(BaseModel):
    """Что принимает API. Pydantic сам проверяет типы и диапазоны."""
    Name: str
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120)
    Pclass: Literal[1, 2, 3]
    SibSp: int = Field(0, ge=0)
    Parch: int = Field(0, ge=0)
    Fare: Optional[float] = Field(None, ge=0)
    Embarked: Optional[Literal["S", "C", "Q"]] = None


class Prediction(BaseModel):
    survived: int
    probability: float


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["titanic"] = load_model()      # модель грузим ОДИН раз при старте
    yield
    models.clear()


app = FastAPI(title="Titanic survival API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(passenger: Passenger):
    return predict_passenger(models["titanic"], passenger.model_dump())