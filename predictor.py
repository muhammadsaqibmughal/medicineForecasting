import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

class MedicineDemandPredictor:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.label_encoders = None
        self.feature_columns = None
        self.daily_data = None
        self.medicines_df = None
        self.pharmacies_df = None
        self.load_model()

    # ---------------------------------------------------------
    # LOAD MODEL + DATA
    # ---------------------------------------------------------
    def load_model(self):
        # Load ML model
        self.model = joblib.load(f"{self.model_path}/XGBoost_model.pkl")

        # Load selected features
        with open(f"{self.model_path}/selected_features.json", "r") as f:
            self.feature_columns = json.load(f)

        # Load data
        self.daily_data = pd.read_csv(f"{self.model_path}/daily_data_engineered.csv")
        self.daily_data["date"] = pd.to_datetime(self.daily_data["date"]).dt.date

        self.medicines_df = pd.read_csv(f"{self.model_path}/medicines_enriched.csv")
        self.pharmacies_df = pd.read_csv(f"{self.model_path}/pharmacies.csv")

        try:
            with open(f"{self.model_path}/label_encoders.pkl", "rb") as f:
                self.label_encoders = joblib.load(f)
        except FileNotFoundError:
            # Recreate encoders from available data
            self.label_encoders = self._recreate_label_encoders()
            with open(f"{self.model_path}/label_encoders.pkl", "wb") as f:
                joblib.dump(self.label_encoders, f)

    # ---------------------------------------------------------
    # LABEL ENCODER RE-CREATION
    # ---------------------------------------------------------
    def _recreate_label_encoders(self):
        label_encoders = {}

        categorical_cols = {
            'category': self.medicines_df.get('category', pd.Series()),
            'therapeuticClass': self.medicines_df.get('therapeuticClass', pd.Series()),
            'seasonalityType': self.medicines_df.get('seasonalityType', pd.Series()),
            'city': self.pharmacies_df.get('city', pd.Series()),
            'state': self.pharmacies_df.get('state', pd.Series()),
            'weatherCondition': self.daily_data.get('weatherCondition', pd.Series())
        }

        for col, series in categorical_cols.items():
            le = LabelEncoder()
            values = series.fillna("Unknown").astype(str).unique()
            le.fit(values)
            label_encoders[col] = le

        return label_encoders

    # ---------------------------------------------------------
    # LIST FUNCTIONS
    # ---------------------------------------------------------
    def list_medicines(self, search_term=None, limit=50):
        df = self.medicines_df
        if search_term:
            df = df[
                df["brandName"].str.contains(search_term, case=False, na=False) |
                df["genericName"].str.contains(search_term, case=False, na=False)
            ]
        return df.head(limit)

    def list_pharmacies(self):
        return self.pharmacies_df

    # ---------------------------------------------------------
    # MAIN PREDICTION FUNCTION
    # ---------------------------------------------------------
    def predict_demand(self, brand_name, prediction_date=None, days_ahead=7, pharmacy_name=None):

        # Find medicine
        medicine = self.medicines_df[
            self.medicines_df["brandName"].str.contains(brand_name, case=False, na=False)
        ]
        if medicine.empty:
            medicine = self.medicines_df[
                self.medicines_df["genericName"].str.contains(brand_name, case=False, na=False)
            ]
        if medicine.empty:
            return None

        medicine = medicine.iloc[0]
        medicine_id = medicine["id"]

        # Pharmacy selection
        if pharmacy_name:
            pharmacy = self.pharmacies_df[
                self.pharmacies_df["pharmacyName"].str.contains(pharmacy_name, case=False, na=False)
            ]
            if pharmacy.empty:
                return None
            pharmacies = [pharmacy.iloc[0]]
        else:
            pharmacies = self.pharmacies_df.to_dict("records")

        # Prediction start date
        if prediction_date is None:
            prediction_date = datetime.now().date() + timedelta(days=1)
        elif isinstance(prediction_date, str):
            prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()

        all_predictions = []

        for pharmacy_data in pharmacies:
            pharmacy_id = pharmacy_data["id"]
            hist_data = self.daily_data[
                (self.daily_data["medicineId"] == medicine_id) &
                (self.daily_data["pharmacyId"] == pharmacy_id)
            ].sort_values("date")

            if hist_data.empty:
                continue

            for d in range(days_ahead):
                date = prediction_date + timedelta(days=d)
                features = self._prepare_features(hist_data, date, medicine, pharmacy_data)

                X = features[self.feature_columns]
                pred = max(0, float(self.model.predict(X)[0]))

                all_predictions.append({
                    "date": date,
                    "day_name": date.strftime("%A"),
                    "medicine": medicine["brandName"],
                    "pharmacy": pharmacy_data["pharmacyName"],
                    "city": pharmacy_data["city"],
                    "predicted_demand": round(pred, 2),
                    "rounded_demand": int(round(pred))
                })

        return pd.DataFrame(all_predictions)

    # ---------------------------------------------------------
    # FEATURE ENGINEERING FOR PREDICTION
    # ---------------------------------------------------------
    def _prepare_features(self, hist, date, medicine, pharmacy):
        last = hist.iloc[-1]
        features = {}

        # Time features
        dt = pd.to_datetime(date)
        features.update({
            "dayOfWeek": dt.dayofweek,
            "month": dt.month,
            "quarter": (dt.month - 1) // 3 + 1,
            "dayOfYear": dt.dayofyear,
            "isWeekend": int(dt.dayofweek >= 5)
        })

        # Medicine attributes
        attributes = ["prescriptionRequired", "isFastMoving", "isEssential"]
        for attr in attributes:
            features[attr] = int(medicine.get(attr, 0))

        # Encoded categorical fields
        for col in ["category", "therapeuticClass", "seasonalityType"]:
            val = str(medicine.get(col, "Unknown"))
            features[f"{col}_enc"] = self._encode(col, val)

        for col in ["city", "state"]:
            val = str(pharmacy.get(col, "Unknown"))
            features[f"{col}_enc"] = self._encode(col, val)

        # Weather
        weather = last.get("weatherCondition", "Clear")
        features["weatherCondition_enc"] = self._encode("weatherCondition", weather)

        # Statistical historical features
        recent = hist.tail(28)["demand"].values
        features["lag_1"] = recent[-1] if len(recent) >= 1 else 0
        features["lag_7"] = recent[-7] if len(recent) >= 7 else 0

        # Rolling windows
        for w in [3, 7, 14, 28]:
            if len(recent) >= w:
                features[f"roll_mean_{w}"] = recent[-w:].mean()
                features[f"roll_std_{w}"] = recent[-w:].std()
            else:
                features[f"roll_mean_{w}"] = 0
                features[f"roll_std_{w}"] = 0

        # Weather numeric
        features["temperature"] = last.get("temperature", 25)
        features["humidity"] = last.get("humidity", 60)

        # Holidays
        features.update({
            "isHoliday": 0,
            "isPublicHoliday": 0,
            "isFestivalSeason": 0
        })

        # Ensure all required features exist
        for col in self.feature_columns:
            features.setdefault(col, 0)

        return pd.DataFrame([features])

    # ---------------------------------------------------------
    # HELPER ENCODER
    # ---------------------------------------------------------
    def _encode(self, column, value):
        try:
            return self.label_encoders[column].transform([value])[0]
        except:
            return 0

