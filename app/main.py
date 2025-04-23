from fastapi import FastAPI, APIRouter, HTTPException
# from loguru import logger
# import mlflow.pyfunc
import pandas as pd
import uvicorn
from Headers import headers

api = APIRouter()
api.include_router(headers.router)

if __name__ == "__main__": #
    uvicorn.run("main:api", host="localhost", port=8000)    #