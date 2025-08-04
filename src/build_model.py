from preprocessing import *
import pandas as pd
import pickle
import xgboost as xgb
import os
os.makedirs("model", exist_ok=True)

# Load data
df_train = pd.read_csv("data/raw_data.csv")

# Preprocess
dp = DataProcessing()
dp.cleaning_steps(df_train)
dp.extract_label_value(df_train)
dp.perform_feature_engineering(df_train)

# Features and label
X = df_train.drop('Time_taken(min)', axis=1)
y = df_train['Time_taken(min)']

# Label encoding
label_encoders = dp.label_encoding(X)
feature_columns = X.columns.tolist()  # Save column order for inference

# Train/test split and scaling
X_train, X_test, y_train, y_test = dp.data_split(X, y)
X_train, X_test, scaler = dp.standardize(X_train, X_test)

# Build model
model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
dp.evaluate_model(y_test, y_pred)

# Save model, encoders, scaler, and feature column order
with open("model/model.pickle", 'wb') as f:
    pickle.dump((model, label_encoders, scaler, feature_columns), f)

print("✅ Model pickle saved to model folder")
