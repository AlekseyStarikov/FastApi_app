from fastapi import APIRouter, HTTPException
import mlflow.pyfunc
import pandas as pd
from utils import logger
from Schema import schema

router=APIRouter()

@router.get("/health")
async def health_check():
    logger.logger.info("Health check requested.")
    return {"status": "OK"}


@router.post("/predict/{model_name}")
async def predict(model_name: str, request: schema.PredictRequest):
    logger.logger.info(f"Prediction requested for model: {model_name}")
    try:
        model_uri = f"models:/{model_name}/latest"
        logger.logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        input_data = pd.DataFrame(request.data)
        logger.logger.debug(f"Input data: {input_data}")

        predictions = model.predict(input_data)
        logger.logger.info("Prediction completed.")

        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
