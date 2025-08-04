from src.preprocessing import DataProcessing
import pickle
import numpy as np

def predict(X):
    # Load model, encoders, scaler, and expected columns
    with open("model/model.pickle", 'rb') as f:
        print("✅ Model imported")
        model, label_encoders, scaler, expected_columns = pickle.load(f)

    # Preprocessing
    dataprocess = DataProcessing()
    X = dataprocess.cleaning_steps(X)
    X = dataprocess.perform_feature_engineering(X)

    # Label encoding
    for column, encoder in label_encoders.items():
        if column in X.columns:
            try:
                X[column] = encoder.transform(X[column])
            except Exception as e:
                print(f"[ERROR] Label encoding failed for column '{column}': {e}")
                raise
        else:
            raise ValueError(f"Expected column '{column}' not found in input")

    # Fill missing columns with 0 (just in case)
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0

    # Reorder to match training
    X = X[expected_columns]

    # Convert to float32
    X = X.astype(np.float32)

    # Scale and predict
    X_scaled = scaler.transform(X)
    print("Input shape:", X_scaled.shape)
    print("Model expects:", model.get_booster().num_features())

    # Final input debug
    print("=== FINAL INPUT CHECK ===")
    print("Data types:\n", X.dtypes)
    print("Nulls:\n", X.isnull().sum())
    print("Shape:", X.shape)
    print("Sample row:\n", X.head(1))

    # Convert and scale
    X = X.astype(np.float32)
    X_scaled = scaler.transform(X)

    # Guard checks
    assert X_scaled.dtype == np.float32, "X_scaled is not float32"
    assert not np.isnan(X_scaled).any(), "NaNs detected in scaled input"
    pred = model.predict(X_scaled)
    return pred
