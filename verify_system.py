import os
import sys
import json
import numpy as np
import cv2

# Ensure we can import modules from the app directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.real_data_trainer import train_and_save_all
from app.utils.tabular_engine import TabularDiagnosticEngine
from app.utils.vision_engine import ECGVisionEngine
from app.utils.llm_engine import ClinicalReportingEngine

def create_synthetic_ecg_image(filepath: str):
    """
    Generates a synthetic ECG report scan image with a pink/red grid paper
    background and a dark black electrocardiogram trace line.
    """
    # 400x400 white canvas
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw pink/red grid lines (grid background)
    # Pink color in BGR format: B=180, G=120, R=255
    grid_color = (180, 120, 255)
    for i in range(0, 400, 15):
        # Horizontal lines
        cv2.line(img, (0, i), (400, i), grid_color, 1)
        # Vertical lines
        cv2.line(img, (i, 0), (i, 400), grid_color, 1)
        
    # Draw a dark black simulated ECG waveform (QRS complexes)
    points = []
    for x in range(0, 400):
        y = 200  # Baseline
        cycle = x % 80
        if 5 <= cycle < 10:
            y = 195  # P wave
        elif 10 <= cycle < 12:
            y = 200
        elif cycle == 15:
            y = 210  # Q wave
        elif cycle == 18:
            y = 130  # R wave peak
        elif cycle == 21:
            y = 240  # S wave dip
        elif 22 <= cycle < 25:
            y = 200
        elif 30 <= cycle < 40:
            y = 188  # T wave
        points.append((x, int(y)))
        
    # Draw lines connecting the points
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (0, 0, 0), 2)
        
    cv2.imwrite(filepath, img)
    print(f"Created synthetic ECG image at {filepath}")

def run_tests():
    print("=== STARTING END-TO-END PIPELINE VALIDATION ===")
    
    assets_dir = "./assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Generate Models and Assets
    print("\n--- Step 1: Generating Model Assets ---")
    train_and_save_all(assets_dir)
    
    # 2. Test Tabular Engine
    print("\n--- Step 2: Testing Tabular Diagnostic Engine ---")
    tab_engine = TabularDiagnosticEngine(model_dir=assets_dir)
    
    # Vitals: Age=58, BP=135, Cholesterol=254, MaxHR=142
    tab_result = tab_engine.preprocess_and_predict(58.0, 135.0, 254.0, 142.0)
    print("Tabular Predictions Results:")
    print(f" - Risk Probability: {tab_result['risk_probability']:.4f}")
    print(f" - SHAP values: {tab_result['shap_values']}")
    print(f" - Aggravating features: {tab_result['aggravating_features']}")
    
    assert 0.0 <= tab_result['risk_probability'] <= 1.0, "Risk probability out of range!"
    assert len(tab_result['shap_values']) == 4, "SHAP values should correspond to all 4 features!"
    print("Tabular Engine OK.")
    
    # 3. Test Vision Engine
    print("\n--- Step 3: Testing ECG Vision Preprocessing & Model Inference ---")
    synthetic_image_path = os.path.join(assets_dir, "synthetic_ecg.png")
    create_synthetic_ecg_image(synthetic_image_path)
    
    cnn_file = os.path.join(assets_dir, "resnet_ecg.weights.h5")
    vis_engine = ECGVisionEngine(model_path=cnn_file)
    
    # Run cleaning and verification
    cleaned_img = vis_engine.clean_ecg_image(synthetic_image_path)
    assert cleaned_img.shape == (224, 224, 3), f"Cleaned image shape mismatch: {cleaned_img.shape}"
    assert cleaned_img.min() >= 0.0 and cleaned_img.max() <= 1.0, "Cleaned image pixels are not normalized!"
    
    # Save processed image to inspect grid line removal
    processed_output_path = os.path.join(assets_dir, "processed_ecg.png")
    # Convert from [0, 1] back to [0, 255] for saving
    cv2.imwrite(processed_output_path, (cleaned_img * 255).astype(np.uint8))
    print(f"Saved processed ECG (grid lines removed) to {processed_output_path} for visual verification")
    
    # Run prediction
    vis_result = vis_engine.predict_ecg(synthetic_image_path)
    print("Vision Prediction Results:")
    print(f" - ECG Risk Probability: {vis_result['ecg_risk_probability']:.4f}")
    print(f" - ECG Classification Status: {vis_result['ecg_status']}")
    print(f" - Cleaned Matrix Shape: {vis_result['processed_shape']}")
    
    assert 0.0 <= vis_result['ecg_risk_probability'] <= 1.0, "ECG risk probability out of range!"
    print("Vision Engine OK.")
    
    # 4. Test Generative AI Reporting Engine
    print("\n--- Step 4: Testing Clinical Reporting Engine ---")
    reporting_engine = ClinicalReportingEngine(api_key=None) # Tests the fallback local engine
    report = reporting_engine.generate_report(tab_result, vis_result)
    print("Clinical Assessment Report Sample Output:")
    print("-" * 50)
    print(report)
    print("-" * 50)
    
    # Assertions
    assert "DISCLAIMER:" in report, "Required clinical disclaimer is missing!"
    assert "Patient Risk Overview" in report, "Report section 'Patient Risk Overview' missing!"
    print("Clinical Reporting Engine OK.")
    
    print("\n=== ALL SYSTEM TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_tests()
