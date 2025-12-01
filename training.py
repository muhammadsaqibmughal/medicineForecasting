# importing all required packages
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib
import json
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine

# ==================== LOAD SYNTHETIC DATA ====================
def load_synthetic_data():
    
    try:
        medicines = pd.read_csv('./medicines_enriched.csv',
                               usecols=['id', 'genericName', 'brandName', 'category', 'therapeuticClass',
                                       'prescriptionRequired', 'seasonalityType', 'isFastMoving', 'isEssential'])

        pharmacies = pd.read_csv('./pharmacies.csv',
                                usecols=['id', 'city', 'state'])

        external_factors = pd.read_csv('./external_factors.csv',
                                      usecols=['date', 'pharmacyId', 'isPublicHoliday', 'isFestivalSeason',
                                              'temperature', 'humidity', 'weatherCondition', 'rainfall', 'airQuality'])

        sales = pd.read_csv('./sales.csv',
                           usecols=['id', 'pharmacyId', 'createdAt', 'dayOfWeek', 'isWeekend',
                                   'isHoliday', 'month', 'quarter', 'weekOfMonth', 'isMonthEnd', 'isMonthStart'])

        sale_items = pd.read_csv('./sale_items.csv',
                                usecols=['saleId', 'pharmacyProductId', 'quantity', 'unitPrice', 'totalPrice', 'discount'])

        sales_patterns = pd.read_csv('./medicine_sales_patterns.csv',
                                    usecols=['medicineId', 'pharmacyId', 'avgDailySales', 'avgWeeklySales',
                                            'medianDailySales', 'peakMonth', 'peakDayOfWeek', 'daysSinceLastSale'])

        print("Synthetic data loaded successfully")
        return {
            'medicines': medicines,
            'pharmacies': pharmacies,
            'external_factors': external_factors,
            'sales': sales,
            'sale_items': sale_items,
            'sales_patterns': sales_patterns
        }
    except FileNotFoundError as e:
        print(f"Error loading synthetic data: {e}")
        return None

# ==================== FETCH REAL DATA FROM DATABASE ====================
def fetch_real_data_from_db(db_uri, pharmacy_id):
    try:
        engine = create_engine(db_uri)
        
        print(f"Fetching real data for pharmacy {pharmacy_id}...")
        
        # Medicines table (all medicines)
        medicines = pd.read_sql(
            f"""
        SELECT m.id, m.genericName, m.brandName, m.category, m.therapeuticClass, 
            m.prescriptionRequired, m.seasonalityType, m.isFastMoving, m.isEssential
        FROM medicines m
        JOIN pharmacy_products pp ON m.id = pp.medicineId
        WHERE pp.pharmacyId = '{pharmacy_id}'
        """, 
            engine
        )
        
        # Specific pharmacy data
        pharmacies = pd.read_sql(
            f"SELECT id, city, state FROM pharmacies WHERE id = '{pharmacy_id}'", 
            engine
        )
        
        if pharmacies.empty:
            raise ValueError(f"Pharmacy {pharmacy_id} not found in database")
        
        # External factors for this pharmacy
        external_factors = pd.read_sql(
            f"SELECT date, pharmacyId, isPublicHoliday, isFestivalSeason, temperature, "
            f"humidity, weatherCondition, rainfall, airQuality "
            f"FROM external_factors WHERE pharmacyId = '{pharmacy_id}'", 
            engine
        )
        
        # Sales for this pharmacy
        sales = pd.read_sql(
            f"SELECT id, pharmacyId, createdAt, dayOfWeek, isWeekend, isHoliday, "
            f"month, quarter, weekOfMonth, isMonthEnd, isMonthStart "
            f"FROM sales WHERE pharmacyId = '{pharmacy_id}'", 
            engine
        )
        
        if sales.empty:
            raise ValueError(f"No sales data found for pharmacy {pharmacy_id}")
        
        # Sale items (join with sales to ensure only items for this pharmacy)
        sale_items = pd.read_sql(
            f"SELECT si.saleId, si.pharmacyProductId, si.quantity, si.unitPrice, "
            f"si.totalPrice, si.discount "
            f"FROM sale_items si "
            f"JOIN sales s ON si.saleId = s.id "
            f"WHERE s.pharmacyId = '{pharmacy_id}'", 
            engine
        )
        
        # Sales patterns for this pharmacy
        sales_patterns = pd.read_sql(
            f"SELECT medicineId, pharmacyId, avgDailySales, avgWeeklySales, "
            f"medianDailySales, peakMonth, peakDayOfWeek, daysSinceLastSale "
            f"FROM medicine_sales_patterns WHERE pharmacyId = '{pharmacy_id}'", 
            engine
        )
        
        print(f" Real data fetched: {len(sales)} sales, {len(sale_items)} sale items")
        
        return {
            'medicines': medicines,
            'pharmacies': pharmacies,
            'external_factors': external_factors,
            'sales': sales,
            'sale_items': sale_items,
            'sales_patterns': sales_patterns
        }
    
    except Exception as e:
        print(f"Error fetching real data: {e}")
        raise

# ==================== VALIDATE REAL DATA ====================
def validate_real_data(real_data):
   
    sales = real_data['sales']
    sale_items = real_data['sale_items']
    
    if sales.empty or sale_items.empty:
        return False, "No sales data available", None
    
    # Convert dates
    sales['createdAt'] = pd.to_datetime(sales['createdAt'])
    
    # Calculate date range
    min_date = sales['createdAt'].min()
    max_date = sales['createdAt'].max()
    date_range_days = (max_date - min_date).days
    
    # Count unique days with sales
    unique_days = sales['createdAt'].dt.date.nunique()
    
    # Count total sales
    total_sales = len(sales)
    total_items = len(sale_items)
    
    # Count unique medicines sold
    sale_items_copy = sale_items.copy()
    sale_items_copy['medicineId'] = sale_items_copy['pharmacyProductId'].apply(
        lambda x: '_'.join(x.split('_')[-2:]) if isinstance(x, str) and len(x.split('_')) > 1 else None
    )
    unique_medicines = sale_items_copy['medicineId'].nunique()
    
    data_stats = {
        'min_date': min_date.date(),
        'max_date': max_date.date(),
        'date_range_days': date_range_days,
        'unique_days_with_sales': unique_days,
        'total_sales': total_sales,
        'total_items': total_items,
        'unique_medicines': unique_medicines,
        'avg_sales_per_day': total_sales / unique_days if unique_days > 0 else 0
    }
    
    print(f"\nReal Data Statistics:")
    print(f"  Date Range: {min_date.date()} to {max_date.date()} ({date_range_days} days)")
    print(f"  Days with Sales: {unique_days}")
    print(f"  Total Sales: {total_sales}")
    print(f"  Total Items Sold: {total_items}")
    print(f"  Unique Medicines: {unique_medicines}")
    print(f"  Avg Sales/Day: {data_stats['avg_sales_per_day']:.1f}")
    
    # Validation rules
    MIN_DAYS_REQUIRED = 30
    MIN_SALES_REQUIRED = 50
    MIN_UNIQUE_DAYS = 20
    
    if date_range_days < MIN_DAYS_REQUIRED:
        return False, f"Insufficient data history. Need at least {MIN_DAYS_REQUIRED} days, found {date_range_days} days", data_stats
    
    if total_sales < MIN_SALES_REQUIRED:
        return False, f"Insufficient sales volume. Need at least {MIN_SALES_REQUIRED} sales, found {total_sales} sales", data_stats
    
    if unique_days < MIN_UNIQUE_DAYS:
        return False, f"Insufficient sales frequency. Need at least {MIN_UNIQUE_DAYS} days with sales, found {unique_days} days", data_stats
    
    if unique_medicines < 5:
        return False, f"Insufficient medicine variety. Need at least 5 different medicines, found {unique_medicines}", data_stats
    
    return True, "Data validation passed", data_stats

# ==================== COMBINE REAL AND SYNTHETIC DATA ====================
def combine_real_and_synthetic_data(real_data, synthetic_data, real_data_ratio=0.3):
    print(f"\nCombining data with {real_data_ratio*100:.0f}% real data, {(1-real_data_ratio)*100:.0f}% synthetic data...")
    
    combined = {}
    
    # For each table, combine real and synthetic
    for key in ['medicines', 'pharmacies', 'external_factors', 'sales', 'sale_items', 'sales_patterns']:
        real_df = real_data[key]
        synthetic_df = synthetic_data[key]
        
        if key in ['medicines', 'pharmacies']:
            # For reference tables, merge and deduplicate
            combined[key] = pd.concat([real_df, synthetic_df]).drop_duplicates(subset=['id'], keep='first')
        
        elif key in ['sales', 'sale_items', 'external_factors', 'sales_patterns']:
            # For transactional data, sample synthetic data based on ratio
            real_count = len(real_df)
            
            if real_count == 0:
                # If no real data for this table, use all synthetic
                combined[key] = synthetic_df
            else:
                # Calculate synthetic sample size
                synthetic_count = int(real_count * (1 - real_data_ratio) / real_data_ratio)
                
                if synthetic_count > len(synthetic_df):
                    synthetic_count = len(synthetic_df)
                
                # Sample synthetic data
                synthetic_sample = synthetic_df.sample(n=synthetic_count, random_state=42) if synthetic_count > 0 else pd.DataFrame()
                
                # Combine
                combined[key] = pd.concat([real_df, synthetic_sample], ignore_index=True)
                
                print(f"  {key}: {len(real_df)} real + {len(synthetic_sample)} synthetic = {len(combined[key])} total")
    
    return combined

# ==================== COMBINE AND PREPROCESS DATA ====================
def combine_and_preprocess_data(data_dict):
    
    sale_items = data_dict['sale_items'].copy()
    sales = data_dict['sales'].copy()
    medicines = data_dict['medicines'].copy()
    pharmacies = data_dict['pharmacies'].copy()
    external_factors = data_dict['external_factors'].copy()
    sales_patterns = data_dict['sales_patterns'].copy()

    # Extract medicineId from pharmacyProductId
    sale_items['medicineId'] = sale_items['pharmacyProductId'].apply(
        lambda x: '_'.join(x.split('_')[-2:]) if isinstance(x, str) and len(x.split('_')) > 1 else None
    )
    sale_items = sale_items[sale_items['medicineId'].notna()].copy()

    # Merge with sales
    combined = sale_items.merge(
        sales[['id', 'pharmacyId', 'createdAt', 'dayOfWeek', 'isWeekend', 'isHoliday',
               'month', 'quarter', 'weekOfMonth', 'isMonthEnd', 'isMonthStart']],
        left_on='saleId', right_on='id', how='inner', suffixes=('', '_drop')
    )
    combined = combined.drop(columns=[col for col in combined.columns if col.endswith('_drop')])

    # Convert dates
    combined['date'] = pd.to_datetime(combined['createdAt']).dt.date
    external_factors['date'] = pd.to_datetime(external_factors['date']).dt.date

    # Merge medicines
    combined = combined.merge(medicines, left_on='medicineId', right_on='id',
                             how='inner', suffixes=('', '_drop'))
    combined = combined.drop(columns=[col for col in combined.columns if col.endswith('_drop')])

    # Merge pharmacies
    combined = combined.merge(pharmacies, left_on='pharmacyId', right_on='id',
                             how='inner', suffixes=('', '_drop'))
    combined = combined.drop(columns=[col for col in combined.columns if col.endswith('_drop')])

    # Merge external factors
    combined = combined.merge(external_factors, on=['date', 'pharmacyId'], how='left',
                             suffixes=('_from_sales', '_from_external'))

    # Merge sales patterns
    combined = combined.merge(sales_patterns, on=['medicineId', 'pharmacyId'], how='left')

    # Fill missing values
    combined['temperature'] = combined['temperature'].fillna(combined['temperature'].median())
    combined['humidity'] = combined['humidity'].fillna(combined['humidity'].median())
    combined['weatherCondition'] = combined['weatherCondition'].fillna('Clear')
    combined['rainfall'] = combined['rainfall'].fillna(0)
    combined['airQuality'] = combined['airQuality'].fillna('Unknown')
    combined['isPublicHoliday'] = combined['isPublicHoliday'].fillna(False).astype(bool)
    combined['isFestivalSeason'] = combined['isFestivalSeason'].fillna(False).astype(bool)

    pattern_cols = ['avgDailySales', 'avgWeeklySales', 'medianDailySales', 'daysSinceLastSale']
    for col in pattern_cols:
        combined[col] = combined[col].fillna(0)

    combined['peakMonth'] = combined['peakMonth'].fillna(combined['month'])
    combined['peakDayOfWeek'] = combined['peakDayOfWeek'].fillna(combined['dayOfWeek'])

    print(f"✓ Combined shape: {combined.shape}")
    return combined

# ==================== AGGREGATE TO DAILY ====================
def aggregate_to_daily(combined_data):
    
    daily_demand = combined_data.groupby(['date', 'medicineId', 'pharmacyId']).agg({
        'quantity': 'sum',
        'totalPrice': 'sum',
        'genericName': 'first',
        'brandName': 'first',
        'category': 'first',
        'therapeuticClass': 'first',
        'prescriptionRequired': 'first',
        'seasonalityType': 'first',
        'isFastMoving': 'first',
        'isEssential': 'first',
        'dayOfWeek': 'first',
        'isWeekend': 'first',
        'isHoliday': 'first',
        'month': 'first',
        'quarter': 'first',
        'isPublicHoliday': 'first',
        'isFestivalSeason': 'first',
        'temperature': 'mean',
        'humidity': 'mean',
        'weatherCondition': 'first',
        'avgDailySales': 'first',
        'avgWeeklySales': 'first',
        'medianDailySales': 'first',
        'peakMonth': 'first',
        'peakDayOfWeek': 'first',
        'city': 'first',
        'state': 'first'
    }).reset_index()

    daily_demand = daily_demand.rename(columns={'quantity': 'demand'})
    daily_demand = daily_demand.sort_values(['medicineId', 'pharmacyId', 'date'])
    return daily_demand

# ==================== FEATURE ENGINEERING ====================
def engineer_features(daily_data):
   
    daily_data = daily_data.sort_values(['medicineId', 'pharmacyId', 'date'])

    # Lag features (1, 3, 7, 14 days)
    for lag in [1, 3, 7, 14]:
        daily_data[f'lag_{lag}'] = daily_data.groupby(['medicineId', 'pharmacyId'])['demand'].shift(lag)

    # Rolling statistics (3, 7, 14, 28 days)
    for window in [3, 7, 14, 28]:
        daily_data[f'roll_mean_{window}'] = daily_data.groupby(['medicineId', 'pharmacyId'])['demand'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        daily_data[f'roll_std_{window}'] = daily_data.groupby(['medicineId', 'pharmacyId'])['demand'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )

    # Exponential weighted moving average
    daily_data['ewm_7'] = daily_data.groupby(['medicineId', 'pharmacyId'])['demand'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean()
    )

    # Temporal features
    daily_data['dayOfYear'] = pd.to_datetime(daily_data['date']).dt.dayofyear
    daily_data['isPeakMonth'] = (daily_data['month'] == daily_data['peakMonth']).astype(int)
    daily_data['isPeakDayOfWeek'] = (daily_data['dayOfWeek'] == daily_data['peakDayOfWeek']).astype(int)

    # Fill NaN in lag/roll features
    lag_cols = [col for col in daily_data.columns if 'lag' in col or 'roll' in col or 'ewm' in col]
    daily_data[lag_cols] = daily_data[lag_cols].fillna(0)

    # Encode categorical variables
    categorical_cols = ['category', 'therapeuticClass', 'seasonalityType', 'weatherCondition', 'city', 'state']
    label_encoders = {}

    for col in categorical_cols:
        daily_data[col] = daily_data[col].fillna('Unknown')
        le = LabelEncoder()
        daily_data[f'{col}_enc'] = le.fit_transform(daily_data[col].astype(str))
        label_encoders[col] = le

    # Convert boolean to int
    bool_cols = ['prescriptionRequired', 'isFastMoving', 'isEssential', 'isWeekend',
                 'isHoliday', 'isPublicHoliday', 'isFestivalSeason']
    for col in bool_cols:
        daily_data[col] = daily_data[col].astype(int)

    return daily_data, label_encoders

# ==================== FEATURE SELECTION ====================
def select_best_features(X_train, y_train, feature_columns, top_n=25):
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Always include critical features
    critical_features = ['lag_1', 'lag_7', 'roll_mean_7', 'avgDailySales', 'month',
                         'dayOfWeek', 'isFastMoving', 'category_enc']

    print("\nTop Features by Importance:")
    print(feature_importances.head(top_n))
    
    # Get top N from importance ranking
    top_from_importance = feature_importances['feature'].head(top_n).tolist()

    # Combine critical + top features (remove duplicates)
    top_features = list(dict.fromkeys(critical_features + top_from_importance))[:top_n]

    return top_features

# ==================== PREPARE TRAINING DATA ====================
def prepare_training_data(data, use_feature_selection=True, top_n_features=25):
    """Split data and select features"""
    # All available features
    all_features = [
        'dayOfWeek', 'month', 'quarter', 'dayOfYear',
        'isWeekend', 'isHoliday', 'isPublicHoliday', 'isFestivalSeason',
        'isPeakMonth', 'isPeakDayOfWeek',
        'prescriptionRequired', 'isFastMoving', 'isEssential',
        'category_enc', 'therapeuticClass_enc', 'seasonalityType_enc',
        'avgDailySales', 'avgWeeklySales', 'medianDailySales',
        'temperature', 'humidity', 'weatherCondition_enc',
        'city_enc', 'state_enc',
        'lag_1', 'lag_3', 'lag_7', 'lag_14',
        'roll_mean_3', 'roll_mean_7', 'roll_mean_14', 'roll_mean_28',
        'roll_std_3', 'roll_std_7', 'roll_std_14', 'roll_std_28',
        'ewm_7'
    ]

    # Filter medicines with sufficient history
    medicine_counts = data.groupby('medicineId').size()
    valid_medicines = medicine_counts[medicine_counts >= 30].index
    data_filtered = data[data['medicineId'].isin(valid_medicines)].copy()

    # Remove rows with NaN
    data_filtered = data_filtered.dropna(subset=all_features + ['demand'])

    # Time-based split (80-20)
    split_date = data_filtered['date'].quantile(0.8)
    train_mask = data_filtered['date'] <= split_date
    test_mask = data_filtered['date'] > split_date

    X_train_full = data_filtered[train_mask][all_features]
    X_test_full = data_filtered[test_mask][all_features]
    y_train = data_filtered[train_mask]['demand']
    y_test = data_filtered[test_mask]['demand']

    # Feature selection
    if use_feature_selection:
        selected_features = select_best_features(X_train_full, y_train, all_features, top_n=top_n_features)
        X_train = X_train_full[selected_features]
        X_test = X_test_full[selected_features]
        feature_columns = selected_features
    else:
        X_train = X_train_full
        X_test = X_test_full
        feature_columns = all_features

    return X_train, X_test, y_train, y_test, feature_columns, data_filtered

# ==================== TRAIN XGBOOST MODEL ====================
def train_xgboost(X_train, X_test, y_train, y_test):
    
    results = {}

    print("\nTraining XGBoost...")

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

    # Fit model
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = np.maximum(xgb_model.predict(X_train), 0)
    y_test_pred = np.maximum(xgb_model.predict(X_test), 0)

    # Compute metrics
    results["XGBoost"] = {
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Train R²": r2_score(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Test R²": r2_score(y_test, y_test_pred),
        "Test MAPE": mean_absolute_percentage_error(y_test, y_test_pred) * 100,
    }

    print(f"Train R²: {results['XGBoost']['Train R²']:.4f}")
    print(f"Test R²: {results['XGBoost']['Test R²']:.4f}")
    print(f"Test MAPE: {results['XGBoost']['Test MAPE']:.2f}%")

    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
    print(f"5-Fold CV R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f}")

    return xgb_model, y_train_pred, y_test_pred, results

# ==================== MAIN TRAINING PIPELINE ====================
def train_model_with_real_data(db_uri, pharmacy_id, real_data_ratio=0.3, save_path="./updated/"):
    try:
        print("="*60)
        print("MEDICINE DEMAND PREDICTION - TRAINING PIPELINE")
        print("="*60)
        
        #  Fetch real data from database
        print("\n[1/9] Fetching real data from database...")
        real_data = fetch_real_data_from_db(db_uri, pharmacy_id)
        
        #  Validate real data
        print("\n[2/9] Validating real data...")
        is_valid, message, data_stats = validate_real_data(real_data)
        
        if not is_valid:
            error_response = {
                "success": False,
                "error": message,
                "data_stats": data_stats,
                "recommendation": "Please collect more sales data before retraining. "
                                 "Minimum requirements: 30 days history, 50 sales, 20 days with sales."
            }
            print(f"\n VALIDATION FAILED: {message}")
            return error_response
        
        print(f"{message}")
        
        #Load synthetic data
        print("\n[3/9] Loading synthetic training data...")
        synthetic_data = load_synthetic_data()
        if synthetic_data is None:
            raise Exception("Failed to load synthetic data")
        
        # Combine real and synthetic data
        print(f"\n[4/9] Combining real and synthetic data...")
        combined_data_dict = combine_real_and_synthetic_data(
            real_data, 
            synthetic_data, 
            real_data_ratio=real_data_ratio
        )
        
        #Preprocess combined data
        print("\n[5/9] Preprocessing combined data...")
        combined_data = combine_and_preprocess_data(combined_data_dict)
        
        #Aggregate to daily level
        print("\n[6/9] Aggregating to daily level...")
        daily_data = aggregate_to_daily(combined_data)
        
        #  Engineer features
        print("\n[7/9] Engineering features...")
        daily_data, label_encoders = engineer_features(daily_data)
        
        # Save engineered data
        engineered_path = f"{save_path}/daily_data_engineered_{pharmacy_id}.csv"
        daily_data.to_csv(engineered_path, index=False)
        print(f" Saved engineered data: {engineered_path}")
        
        # Step 8: Prepare training data
        print("\n[8/9] Preparing training data...")
        X_train, X_test, y_train, y_test, feature_columns, filtered_data = prepare_training_data(
            daily_data,
            use_feature_selection=True,
            top_n_features=25
        )
        
        # Step 9: Train model
        print("\n[9/9] Training XGBoost model...")
        xgb_model, y_train_pred, y_test_pred, results = train_xgboost(
            X_train, X_test, y_train, y_test
        )
        
        # Save model and features
        print("\n" + "="*60)
        print("SAVING MODEL AND ARTIFACTS")
        print("="*60)
        
        model_filename = f"{save_path}/XGBoost_model_{pharmacy_id}.pkl"
        features_filename = f"{save_path}/selected_features_{pharmacy_id}.json"
        encoders_filename = f"{save_path}/label_encoders_{pharmacy_id}.pkl"
        metadata_filename = f"{save_path}/training_metadata_{pharmacy_id}.json"
        
        # Save model
        joblib.dump(xgb_model, model_filename)
        print(f" Saved model: {model_filename}")
        
        # Save features
        with open(features_filename, 'w') as f:
            json.dump(feature_columns, f, indent=4)
        print(f"Saved features: {features_filename}")
        
        # Save encoders
        joblib.dump(label_encoders, encoders_filename)
        print(f" Saved encoders: {encoders_filename}")
        
        # Save metadata
        metadata = {
            "pharmacy_id": pharmacy_id,
            "training_date": datetime.now().isoformat(),
            "real_data_ratio": real_data_ratio,
            "data_stats": data_stats,
            "model_metrics": results["XGBoost"],
            "feature_count": len(feature_columns),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "medicines_count": filtered_data['medicineId'].nunique(),
            "date_range": {
                "min": str(filtered_data['date'].min()),
                "max": str(filtered_data['date'].max())
            }
        }
        
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f" Saved metadata: {metadata_filename}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nModel Performance:")
        print(f"  - Test R²: {results['XGBoost']['Test R²']:.4f}")
        print(f"  - Test MAPE: {results['XGBoost']['Test MAPE']:.2f}%")
        print(f"  - Test MAE: {results['XGBoost']['Test MAE']:.2f}")
        
        return {
            "success": True,
            "message": "Model trained successfully",
            "metadata": metadata,
            "model_path": model_filename,
            "features_path": features_filename,
            "encoders_path": encoders_filename
        }
        
    except Exception as e:
        print(f"\n TRAINING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== EXAMPLE USAGE ====================
def main():
    # Configuration
    DB_URI = "postgresql://postgres.tyteorafrqsefmfmccvx:PharmaConnect123@aws-1-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"
    PHARMACY_ID = "c04b30dc-d4cd-4d33-bd63-ba92e01bbdb9"
    REAL_DATA_RATIO = 0.3  
    SAVE_PATH = "./updated/"
    
    # Run training
    result = train_model_with_real_data(
        db_uri=DB_URI,
        pharmacy_id=PHARMACY_ID,
        real_data_ratio=REAL_DATA_RATIO,
        save_path=SAVE_PATH
    )
    
    # Print result
    if result["success"]:
        print("\nTraining completed successfully!")
        print(f"Model saved to: {result['model_path']}")
    else:
        print(f"\nTraining failed: {result['error']}")
        
    return result

if __name__ == "__main__":
    # For backward compatibility - train with synthetic data only
    print("Training with synthetic data only...")
    
    # Load synthetic data
    data_dict = load_synthetic_data()
    if data_dict is None:
        print("Failed to load data")
        exit(1)
    
    # Preprocess
    print("\nCombining and preprocessing data...")
    combined_data = combine_and_preprocess_data(data_dict)
    
    # Aggregate to daily
    print("\n Aggregating to daily level...")
    daily_data = aggregate_to_daily(combined_data)
    
    # Engineer features
    print("\ Engineering features...")
    daily_data, label_encoders = engineer_features(daily_data)
    daily_data.to_csv('./updated/daily_data_engineered.csv', index=False)
    
    # Prepare training data
    print("\n Preparing training data...")
    X_train, X_test, y_train, y_test, feature_columns, filtered_data = prepare_training_data(
        daily_data,
        use_feature_selection=True,
        top_n_features=25
    )
    
    # Train models
    print("\nTraining models...")
    xgb_model, y_train_pred, y_test_pred, results = train_xgboost(X_train, X_test, y_train, y_test)
    
    # Save model and features
    print("\nSaving model and results...")
    model_filename = "./updated/XGBoost_model.pkl"
    features_filename = "./updated/selected_features.json"
    
    joblib.dump(xgb_model, model_filename)
    print(f"Saved model: {model_filename}")
    
    with open(features_filename, 'w') as f:
        json.dump(feature_columns, f, indent=4)
    print(f"Saved selected features: {features_filename}")
    
    print("\nTraining completed!")