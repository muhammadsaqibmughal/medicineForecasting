import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Any

class DiseasePredictor:

    def __init__(self, model_filename: str = "disease_model.pkl"):
        self.model = None
        self.symptoms_list = None
        self.disease_list = None
        self.recommendations = None
        self.model_name = None
        
        self._load_model_data(model_filename)

    def _load_model_data(self, filename: str):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            self.model = data.get("model")
            self.symptoms_list = data.get("symptoms_list")
            self.disease_list = data.get("disease_list")
            self.recommendations = data.get("recommendations")
            self.model_name = data.get("model_name", "Unknown Model")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _symptoms_to_vector(self, symptoms: List[str]) -> pd.DataFrame:
        if self.symptoms_list is None:
            return pd.DataFrame()

        input_vector = {symptom: 0 for symptom in self.symptoms_list}
        
        for symptom in symptoms:
            if symptom in input_vector:
                input_vector[symptom] = 1

        return pd.DataFrame([input_vector])

    def predict_disease(self, raw_symptoms: List[str]) -> Dict[str, Any]:
        
        if self.model is None:
            return {
                "success": False,
                "message": "Model not loaded.",
                "prediction": None,
                "recommendations": None,
            }

        if not raw_symptoms:
            return {
                "success": False,
                "message": "No symptoms provided.",
                "prediction": None,
                "recommendations": None,
            }

        input_df = self._symptoms_to_vector(raw_symptoms)

        if input_df.empty:
            return {
                "success": False,
                "message": "Symptoms do not match model features.",
                "prediction": None,
                "recommendations": None,
            }

        # Predict
        predicted_disease = self.model.predict(input_df)[0]

        # Confidence
        proba = self.model.predict_proba(input_df)
        idx = np.where(self.model.classes_ == predicted_disease)[0][0]
        confidence = float(proba[0][idx])

        recs = self.recommendations.get(
            predicted_disease,
            {"message": "No recommendations found."}
        )

        return {
            "success": True,
            "prediction": predicted_disease,
            "confidence": round(confidence * 100, 2),
            "recommendations": recs,
            "model_used": self.model_name
        }
