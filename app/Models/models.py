from pydantic import BaseModel

# Пример входных данных
class PredictRequest(BaseModel):
    data: list