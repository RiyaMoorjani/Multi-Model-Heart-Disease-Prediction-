import os
import json
import numpy as np
import xgboost as xgb
import shap
from typing import Dict, List, Any, Tuple

class TabularDiagnosticEngine:
    def __init__(self, model_dir: str = "."):
        self.model_dir = model_dir
        self.scaler_path = os.path.join(model_dir, "scaler_params.json")
        self.xgb_path = os.path.join(model_dir, "xgboost_model.json")
        self.bg_path = os.path.join(model_dir, "background_data.json")
        
        # Load scaler parameters
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler parameters not found at {self.scaler_path}")
        with open(self.scaler_path, "r") as f:
            self.scaler_params = json.load(f)
            
        self.means = np.array(self.scaler_params["means"], dtype=np.float32)
        self.variances = np.array(self.scaler_params["variances"], dtype=np.float32)
        self.feature_names = self.scaler_params["feature_names"]
        
        # Load XGBoost model
        if not os.path.exists(self.xgb_path):
            raise FileNotFoundError(f"XGBoost model not found at {self.xgb_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.xgb_path)
        
        # Load background data for SHAP
        if not os.path.exists(self.bg_path):
            raise FileNotFoundError(f"SHAP background data not found at {self.bg_path}")
        with open(self.bg_path, "r") as f:
            self.bg_data = np.array(json.load(f), dtype=np.float32)
            
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model, data=self.bg_data)
        
    def preprocess_and_predict(self, age: float, trestbps: float, chol: float, thalach: float) -> Dict[str, Any]:
        """
        Preprocesses continuous patient vitals, computes the heart disease risk probability,
        and extracts features that significantly aggravate risk (SHAP value > 0.05).
        """
        # Formulate input vector [Age, RestingBP, Cholesterol, MaxHR]
        raw_input = np.array([[age, trestbps, chol, thalach]], dtype=np.float32)
        
        # Explicit Standardization: z = (x - mean) / sqrt(variance)
        epsilon = 1e-8
        scaled_input = (raw_input - self.means) / np.sqrt(self.variances + epsilon)
        
        # Compute final prediction probability
        prob = float(self.model.predict_proba(scaled_input)[0, 1])
        
        # Calculate local SHAP values
        shap_vals = self.explainer.shap_values(scaled_input)
        
        # Handle list/multi-class outputs in TreeExplainer
        if isinstance(shap_vals, list):
            # Index 1 represents positive class (disease present)
            shap_vals = shap_vals[1]
            
        # If output is 2D (e.g. [[val1, val2, ...]]), squeeze to 1D
        if len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]
            
        # Map feature names to SHAP values
        shap_dict = {}
        aggravating_features = []
        
        # Define pretty names for prompt usage
        pretty_names = {
            "Age": "Age",
            "RestingBP": "Resting Blood Pressure",
            "Cholesterol": "Cholesterol Level",
            "MaxHR": "Maximum Heart Rate"
        }
        
        for name, val in zip(self.feature_names, shap_vals):
            val_float = float(val)
            shap_dict[name] = val_float
            if val_float > 0.05:
                aggravating_features.append({
                    "feature": name,
                    "pretty_name": pretty_names.get(name, name),
                    "shap_value": val_float,
                    "status": "Aggravating Risk Factor"
                })
                
        return {
            "risk_probability": prob,
            "shap_values": shap_dict,
            "aggravating_features": aggravating_features
        }
