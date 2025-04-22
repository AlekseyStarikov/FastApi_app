from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import mlflow.pyfunc
import pandas as pd
import uvicorn

app = FastAPI()

# Настройка логирования
logger.add("logs/app.log", rotation="500 KB", retention="10 days", level="INFO")

# Пример входных данных
class PredictRequest(BaseModel):
    data: list


@app.get("/")#health
async def health_check():
    logger.info("Health check requested.")
    return {"status": "OK"}


@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictRequest):
    logger.info(f"Prediction requested for model: {model_name}")
    try:
        model_uri = f"models:/{model_name}/latest"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        input_data = pd.DataFrame(request.data)
        logger.debug(f"Input data: {input_data}")

        predictions = model.predict(input_data)
        logger.info("Prediction completed.")

        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8001)    