import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import xgboost as xgb
import joblib
import json
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine


class MedicineDemandTrainer:

    def __init__(self, db_uri=None, pharmacy_id=None, real_data_ratio=0.3, save_path="./updated/"):
        self.db_uri = db_uri
        self.pharmacy_id = pharmacy_id
        self.real_data_ratio = real_data_ratio
        self.save_path = save_path


    # SYNTHETIC DATA LOADER

    def load_synthetic_data(self):
        try:
            medicines = pd.read_csv('./medicines_enriched.csv')
            pharmacies = pd.read_csv('./pharmacies.csv')
            external_factors = pd.read_csv('./external_factors.csv')
            sales = pd.read_csv('./sales.csv')
            sale_items = pd.read_csv('./sale_items.csv')
            sales_patterns = pd.read_csv('./medicine_sales_patterns.csv')

            return {
                'medicines': medicines,
                'pharmacies': pharmacies,
                'external_factors': external_factors,
                'sales': sales,
                'sale_items': sale_items,
                'sales_patterns': sales_patterns
            }

        except Exception as e:
            print("Error loading synthetic data:", e)
            return None

    # REAL DATA FETCHING
    def fetch_real_data(self):
        engine = create_engine(self.db_uri)
        pid = self.pharmacy_id

        def safe_query(query):
            try:
                return pd.read_sql(query, engine)
            except Exception as e:
                print(f"Warning: query failed: {e}")
                return pd.DataFrame()

    # Medicines
        medicines = safe_query(f"""
            SELECT m.* 
            FROM medicines m
            JOIN pharmacy_products pp 
                ON m.id = pp."medicineId"
            WHERE pp."pharmacyId" = '{pid}'
        """)

    # Pharmacies
        pharmacies = safe_query(f"""
            SELECT id, city, state 
            FROM pharmacies 
            WHERE id = '{pid}'
        """)

    # External Factors
        external_factors = safe_query(f"""
            SELECT * 
            FROM external_factors 
            WHERE "pharmacyId" = '{pid}'
        """)

    # Sales
        sales = safe_query(f"""
            SELECT * 
            FROM sales 
            WHERE "pharmacyId" = '{pid}'
        """)

    # Sale Items
        sale_items = safe_query(f"""
            SELECT si.* 
            FROM sale_items si
            JOIN sales s ON si."saleId" = s.id
            WHERE s."pharmacyId" = '{pid}'
        """)

    # Medicine Sales Patterns
        sales_patterns = safe_query(f"""
            SELECT * 
            FROM medicine_sales_patterns 
            WHERE "pharmacyId" = '{pid}'
        """)

        return {
            'medicines': medicines,
            'pharmacies': pharmacies,
            'external_factors': external_factors,
            'sales': sales,
            'sale_items': sale_items,
            'sales_patterns': sales_patterns
        }

    # VALIDATION
    def validate_real_data(self, real):
        sales = real['sales']
        items = real['sale_items']

        # Base validation
        if sales.empty or items.empty:
            return False, {
                "message": "No sales data found in database.",
                "details": {
                    "sales_rows": len(sales),
                    "items_rows": len(items),
                    "note": "Both sales and sale_items tables must contain data."
                }
            }, None

        # Convert dates
        sales['createdAt'] = pd.to_datetime(sales['createdAt'])

        min_date = sales['createdAt'].min()
        max_date = sales['createdAt'].max()
        date_range = (max_date - min_date).days
        unique_days = sales['createdAt'].dt.date.nunique()

        # Extract medicine ID
        items['medicineId'] = items['pharmacyProductId'].apply(
            lambda x: '_'.join(x.split('_')[-2:])
        )
        unique_meds = items['medicineId'].nunique()

        # Stats collected from DB
        stats = {
            "min_date": str(min_date.date()),
            "max_date": str(max_date.date()),
            "date_range_days": date_range,
            "unique_sales_days": unique_days,
            "total_sales_records": len(sales),
            "total_item_records": len(items),
            "unique_medicines_count": unique_meds
        }

        # Validation conditions with improved error messages
        if date_range < 30:
            return False, {
                "message": "Insufficient date range for training.",
                "expected": "At least 30 days between first and last sale.",
                "actual": f"{date_range} days available.",
                "stats": stats
            }, stats

        if len(sales) < 50:
            return False, {
                "message": "Not enough total sales records.",
                "expected": "At least 50 sales records.",
                "actual": f"{len(sales)} sales found.",
                "stats": stats
            }, stats

        if unique_days < 20:
            return False, {
                "message": "Sales data must span across at least 20 unique days.",
                "expected": ">= 20 unique days with sales.",
                "actual": f"{unique_days} unique days found.",
                "stats": stats
            }, stats

        if unique_meds < 5:
            return False, {
                "message": "Not enough diversity in medicines.",
                "expected": "At least 5 unique medicines in sale_items.",
                "actual": f"{unique_meds} unique medicines found.",
                "stats": stats
            }, stats

        # All validations passed
        return True, {
            "message": "Validation passed. Data is sufficient for training.",
            "stats": stats
        }, stats

    # COMBINE REAL + SYNTHETIC
    def combine_real_and_synthetic(self, real, synthetic):
        ratio = self.real_data_ratio
        combined = {}

        for key in real:
            r = real[key]
            s = synthetic[key]

            if key in ['medicines', 'pharmacies']:
                combined[key] = pd.concat([r, s]).drop_duplicates('id')
            else:
                real_n = len(r)
                syn_n = int(real_n * (1 - ratio) / ratio)

                syn_sample = s.sample(min(syn_n, len(s)), random_state=42)

                combined[key] = pd.concat([r, syn_sample], ignore_index=True)

        return combined


    # MERGE & PREPROCESS
    def preprocess(self, data):
        sale_items = data['sale_items'].copy()
        sales = data['sales'].copy()
        meds = data['medicines'].copy()
        pharm = data['pharmacies'].copy()
        external = data['external_factors'].copy()
        patt = data['sales_patterns'].copy()

        sale_items['medicineId'] = sale_items['pharmacyProductId'].apply(
            lambda x: '_'.join(x.split('_')[-2:])
        )

        merged = sale_items.merge(
            sales, left_on='saleId', right_on='id', how='inner'
        )

        merged['date'] = pd.to_datetime(merged['createdAt']).dt.date
        external['date'] = pd.to_datetime(external['date']).dt.date

        merged = merged.merge(meds, left_on='medicineId', right_on='id')
        merged = merged.merge(pharm, left_on='pharmacyId', right_on='id')
        merged = merged.merge(external, on=['date', 'pharmacyId'], how='left')
        merged = merged.merge(patt, on=['medicineId', 'pharmacyId'], how='left')

        merged.fillna(0, inplace=True)

        return merged

    # DAILY AGG
    def daily_aggregate(self, combined):
        daily = combined.groupby(
            ['date', 'medicineId', 'pharmacyId']
        ).agg({
            'quantity': 'sum',
            'totalPrice': 'sum',
            'dayOfWeek': 'first',
            'month': 'first',
            'quarter': 'first',
            'temperature': 'mean',
            'humidity': 'mean',
            'avgDailySales': 'first',
            'avgWeeklySales': 'first',
            'medianDailySales': 'first'
        }).reset_index()

        daily = daily.rename(columns={'quantity': 'demand'})
        return daily

    # FEATURE ENGINEERING
    def engineer_features(self, df):
        df = df.sort_values(['medicineId', 'pharmacyId', 'date'])

        for lag in [1, 3, 7, 14]:
            df[f'lag_{lag}'] = df.groupby(['medicineId', 'pharmacyId'])['demand'].shift(lag)

        for win in [3, 7, 14, 28]:
            df[f'roll_mean_{win}'] = df.groupby(['medicineId', 'pharmacyId'])['demand'] \
                .transform(lambda x: x.rolling(win, 1).mean())

        df.fillna(0, inplace=True)
        return df


    # FEATURE SELECTION
    def select_features(self, X, y, features, top_n=25):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance['feature'].head(top_n).tolist()


    # TRAINING DATA SPLIT
    def prepare_training_data(self, df):
        FEATURES = [col for col in df.columns if 'lag_' in col or 'roll_' in col]

        df = df.dropna(subset=FEATURES + ['demand'])

        split = df['date'].quantile(0.8)
        train = df[df['date'] <= split]
        test = df[df['date'] > split]

        X_train = train[FEATURES]
        X_test = test[FEATURES]
        y_train = train['demand']
        y_test = test['demand']

        selected = self.select_features(X_train, y_train, FEATURES)
        return (
            X_train[selected],
            X_test[selected],
            y_train, y_test,
            selected
        )


    # TRAIN XGBOOST
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = np.maximum(model.predict(X_test), 0)

        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
            "MAPE": mean_absolute_percentage_error(y_test, preds) * 100
        }

        return model, metrics

    # MAIN PIPELINE RUNNER
    def run(self):
        try:
            # Step 1: Load real data
            real = self.fetch_real_data()

            # Step 2: Validate
            ok, msg, stats = self.validate_real_data(real)
            if not ok:
                return {"success": False, "error": msg, "data_stats": stats}

            # Step 3: Load synthetic
            synthetic = self.load_synthetic_data()

            # Step 4: Combine
            combined_dict = self.combine_real_and_synthetic(real, synthetic)

            # Step 5: Preprocess
            combined = self.preprocess(combined_dict)

            # Step 6: Daily aggregation
            daily = self.daily_aggregate(combined)

            # Step 7: Feature engineering
            daily = self.engineer_features(daily)

            # Step 8: Prep training data
            X_train, X_test, y_train, y_test, selected = self.prepare_training_data(daily)

            # Step 9: Train model
            model, metrics = self.train_xgboost(X_train, X_test, y_train, y_test)

            # Save model
            model_path = f"{self.save_path}/model_{self.pharmacy_id}.pkl"
            joblib.dump(model, model_path)

            return {
                "success": True,
                "message": "Model trained successfully",
                "metrics": metrics,
                "model_path": model_path
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
