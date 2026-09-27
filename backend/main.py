"""Run from backend/: uvicorn main:app --reload"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from inference import predict_machine_health


logger = logging.getLogger(__name__)
app = FastAPI(
    title="Predictive Maintenance API",
    description="Frozen model estimates using each machine's stored history.",
)

origins = [
    "http://localhost:5173",
    "https://predictive-machine-maintainance.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class MachineReading(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    machine_id: int
    volt: float
    rotate: float
    pressure: float
    vibration: float


@app.get("/")
def root():
    return {"message": "Predictive Maintenance backend is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(reading: MachineReading):
    try:
        return predict_machine_health(
            machine_id=reading.machine_id,
            volt=reading.volt,
            rotate=reading.rotate,
            pressure=reading.pressure,
            vibration=reading.vibration,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediction failed for machine %s", reading.machine_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs for details.",
        ) from error
