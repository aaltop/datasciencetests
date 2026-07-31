from fastapi import FastAPI
from pydantic import BaseModel

from src import models

app = FastAPI()

model = models.sentence_transformer()


class PredictRequestBody(BaseModel):
    text_docs: list[str]


class PredictResponse(BaseModel):
    response: list[list[float]]


@app.post("/predict")
def predict(request: PredictRequestBody):

    if len(request.text_docs) < 1:
        return PredictResponse(response=[[]])

    encodings = model.encode(request.text_docs)
    return PredictResponse(response=encodings.tolist())
