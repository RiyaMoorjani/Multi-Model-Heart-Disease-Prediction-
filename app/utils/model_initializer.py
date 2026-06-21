import os
import json
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models

def generate_assets(target_dir: str = "."):
    os.makedirs(target_dir, exist_ok=True)
    
    scaler_path = os.path.join(target_dir, "scaler_params.json")
    xgb_path = os.path.join(target_dir, "xgboost_model.json")
    bg_path = os.path.join(target_dir, "background_data.json")
    cnn_path = os.path.join(target_dir, "resnet_ecg.weights.h5")
    
    # 1. Generate Standard Scaler Parameters
    # Features: [Age, Blood Pressure, Cholesterol, Max Heart Rate]
    means = [54.0, 131.0, 246.0, 149.0]
    stds = [9.0, 17.5, 51.5, 22.9]
    variances = [float(s ** 2) for s in stds]
    
    scaler_data = {
        "means": means,
        "variances": variances,
        "feature_names": ["Age", "RestingBP", "Cholesterol", "MaxHR"]
    }
    
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=4)
    print(f"Saved scaler parameters to {scaler_path}")
    
    # 2. Generate Synthetic Tabular Data & Train XGBoost
    np.random.seed(42)
    n_samples = 200
    
    # Generate random features based on distributions
    age = np.random.normal(means[0], stds[0], n_samples)
    bps = np.random.normal(means[1], stds[1], n_samples)
    chol = np.random.normal(means[2], stds[2], n_samples)
    mhr = np.random.normal(means[3], stds[3], n_samples)
    
    X = np.stack([age, bps, chol, mhr], axis=1)
    
    # Standardize
    X_scaled = (X - means) / np.sqrt(variances)
    
    # Clinical probability logic (risk increases with age, BP, chol; decreases with Max HR)
    logits = 0.04 * X_scaled[:, 0] + 0.03 * X_scaled[:, 1] + 0.02 * X_scaled[:, 2] - 0.05 * X_scaled[:, 3] - 0.5
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (prob > np.random.uniform(0, 1, n_samples)).astype(int)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=15,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="binary:logistic"
    )
    xgb_model.fit(X_scaled, y)
    xgb_model.save_model(xgb_path)
    print(f"Saved trained XGBoost model to {xgb_path}")
    
    # 3. Save background dataset for SHAP explainability
    # Use first 50 samples of scaled data as representative background
    bg_data = X_scaled[:50].tolist()
    with open(bg_path, "w") as f:
        json.dump(bg_data, f)
    print(f"Saved SHAP background data to {bg_path}")
    
    # 4. Generate Simple CNN for ECG (Simulating ResNet/CNN - without BatchNormalization to ensure cross-version compatibility)
    inputs = layers.Input(shape=(224, 224, 3))
    
    # Initial block
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual-like Convolution Block
    residual = layers.Conv2D(32, (1, 1), strides=(2, 2), padding="same")(x) # projection shortcut
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.add([x, residual])
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    cnn_model = models.Model(inputs=inputs, outputs=outputs)
    cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Save the TensorFlow model weights as H5 file
    cnn_model.save_weights(cnn_path)
    print(f"Saved dummy ResNet ECG model weights to {cnn_path}")

if __name__ == "__main__":
    generate_assets(".")
