# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional, List
# import uvicorn
# import pandas as pd
# from fastapi.responses import JSONResponse

# from predictor import MedicineDemandPredictor
# from Training2 import MedicineDemandTrainer


# # Configuration

# DB_URI = "postgresql://postgres.tyteorafrqsefmfmccvx:PharmaConnect123@aws-1-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"
# SAVE_PATH = "./updated/"
# # pharmacy_id = "c04b30dc-d4cd-4d33-bd63-ba92e01bbdb9";
# # real_data_ratio = 0.3;

# # Initialize FastAPI

# app = FastAPI(
#     title="Medicine Demand Prediction API",
#     version="1.0.0",
#     description="FastAPI backend for predicting medicine demand using ML"
# )


# # Load Predictor & Trainer

# MODEL_PATH = "./"
# predictor = MedicineDemandPredictor(model_path=MODEL_PATH)

# trainer = MedicineDemandTrainer(
#     db_uri=DB_URI,
#     save_path=SAVE_PATH
# )



# # API REQUEST MODELS

# class PredictionRequest(BaseModel):
#     medicine: List[str]
#     prediction_date: Optional[str] = None
#     days_ahead: int = 7
#     pharmacy_name: Optional[str] = None


# class TrainRequest(BaseModel):
#     pharmacy_id: str
#     real_data_ratio: float = 0.3


# # ------------------------------
# # ROUTES
# # ------------------------------

# @app.get("/health")
# def health_check():
#     return {"status": "running", "message": "Medicine Demand API is active"}


# @app.get("/list-pharmacies")
# def list_pharmacies():
#     df = predictor.list_pharmacies()
#     return df.to_dict(orient="records")



# @app.post("/predict")
# def predict_demand(request: PredictionRequest):
#     all_predictions = []

#     for med in request.medicine:
#         try:
#             df = predictor.predict_demand(
#                 brand_name=med,
#                 prediction_date=request.prediction_date,
#                 days_ahead=request.days_ahead,
#                 pharmacy_name=request.pharmacy_name
#             )
#             if df is not None and not df.empty:
#                 all_predictions.extend(df.to_dict(orient="records"))
#         except Exception as e:
#             print(f"Error predicting for medicine '{med}': {e}")

#     if not all_predictions:
#         return JSONResponse(
#             status_code=200,
#             content={"success": False, "message": "No predictions available"}
#         )

#     return {"success": True, "predictions": all_predictions}


# @app.post("/train")
# def train_model(request: TrainRequest):
#     trainer = MedicineDemandTrainer(
#         db_uri=DB_URI,
#         pharmacy_id=request.pharmacy_id,
#         real_data_ratio=request.real_data_ratio,
#         save_path=SAVE_PATH
#     )
#     result = trainer.run()
#     return result

# # ------------------------------
# # Run Server
# # ------------------------------
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import pandas as pd
from fastapi.responses import JSONResponse

from diseasePredictor import DiseasePredictor 
from predictor import MedicineDemandPredictor
from Training2 import MedicineDemandTrainer


# --- Configuration ---
DB_URI = "postgresql://postgres.tyteorafrqsefmfmccvx:PharmaConnect123@aws-1-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"
SAVE_PATH = "./updated/"
MODEL_PATH = "./" 
# # pharmacy_id = "c04b30dc-d4cd-4d33-bd63-ba92e01bbdb9";
# # real_data_ratio = 0.3;



app = FastAPI(
    title="PharmaConnect+",
    version="1.0.0",
    description="FastAPI backend for predicting medicine demand and diagnosing diseases."
)

# --- Load Predictors & Trainers ---


predictor_demand = MedicineDemandPredictor(model_path=MODEL_PATH) 
trainer_demand = MedicineDemandTrainer(db_uri=DB_URI, save_path=SAVE_PATH)

# Load Disease Predictor
DISEASE_MODEL_FILE = "disease_model.pkl"
disease_predictor = None
if os.path.exists(DISEASE_MODEL_FILE):
    try:
        disease_predictor = DiseasePredictor(model_filename=DISEASE_MODEL_FILE)
    except Exception as e:
        print(f"Error loading Disease Predictor: {e}")
        disease_predictor = None
else:
    print(f"WARNING: Disease model '{DISEASE_MODEL_FILE}' not found.")





class PredictionRequest(BaseModel):
    medicine: List[str]
    prediction_date: Optional[str] = None
    days_ahead: int = 7
    pharmacy_name: Optional[str] = None

class TrainRequest(BaseModel):
    pharmacy_id: str
    real_data_ratio: float = 0.3


class SymptomDiagnosisRequest(BaseModel):
    symptoms: List[str]

class DiagnosisResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[str] = None
    recommendations: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    message: Optional[str] = None

@app.get("/health")
def health_check():
    return {
        "status": "running", 
        "medicine_demand_active": True, 
        "disease_diagnosis_active": disease_predictor is not None
    }


@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose_disease(request: SymptomDiagnosisRequest):
    if disease_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Disease prediction service is unavailable. Model file not found or failed to load."
        )

    input_symptoms = [s.lower() for s in request.symptoms]
    result = disease_predictor.predict_disease(input_symptoms)
    return result




@app.post("/predict")
def predict_demand(request: PredictionRequest):
    all_predictions = []

    for med in request.medicine:
        try:
            df = predictor.predict_demand(
                brand_name=med,
                prediction_date=request.prediction_date,
                days_ahead=request.days_ahead,
                pharmacy_name=request.pharmacy_name
            )
            if df is not None and not df.empty:
                all_predictions.extend(df.to_dict(orient="records"))
        except Exception as e:
            print(f"Error predicting for medicine '{med}': {e}")

    if not all_predictions:
        return JSONResponse(
            status_code=200,
            content={"success": False, "message": "No predictions available"}
        )

    return {"success": True, "predictions": all_predictions}


@app.post("/train")
def train_model(request: TrainRequest):
    trainer = MedicineDemandTrainer(
        db_uri=DB_URI,
        pharmacy_id=request.pharmacy_id,
        real_data_ratio=request.real_data_ratio,
        save_path=SAVE_PATH
    )
    result = trainer.run()
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)