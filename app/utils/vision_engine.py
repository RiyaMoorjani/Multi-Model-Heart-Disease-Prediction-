import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Union

def build_ecg_model() -> tf.keras.Model:
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    residual = layers.Conv2D(32, (1, 1), strides=(2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.add([x, residual])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

class ECGVisionEngine:
    def __init__(self, model_path: str = "./resnet_ecg.weights.h5"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ECG CNN model weights not found at {self.model_path}")
        self.model = build_ecg_model()
        self.model.load_weights(self.model_path)

    def clean_ecg_image(self, image_content: Union[str, bytes]) -> np.ndarray:
        """
        Cleans an ECG image report by converting it to grayscale, applying Gaussian smoothing,
        and using adaptive thresholding to eliminate pink/red background paper grids.
        Formats and returns the image as a (224, 224, 3) normalized float32 matrix.
        """
        if isinstance(image_content, bytes):
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_content, cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError("Failed to load image. Ensure it is a valid visual format.")
            
        # Convert to HSV to separate pink/red grid lines
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect pink/red grids: Hue ranges [0, 10] and [170, 180]
        lower_red1 = np.array([0, 30, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 30, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        grid_mask = cv2.bitwise_or(mask1, mask2)
        
        # Replace detected pink/red grid lines with pure white background pixels
        cleaned = img.copy()
        cleaned[grid_mask > 0] = [255, 255, 255]
        
        # Convert to Grayscale
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        
        # Gaussian smoothing to reduce sensor/paper grain noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Thresholding to extract dark waveform traces
        # Using binary inversion so the trace lines are represented as high values (255)
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Morphological opening (erosion followed by dilation) to clean minor isolated noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Resize to target matrix size (224, 224)
        resized = cv2.resize(morph, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert single channel grayscale back to 3-channel (RGB) format
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Standardize pixel values to [0.0, 1.0] range
        normalized = rgb_image.astype(np.float32) / 255.0
        
        return normalized

    def predict_ecg(self, image_content: Union[str, bytes]) -> Dict[str, Any]:
        """
        Preprocesses raw ECG image, runs model inference, and returns risk analysis details.
        """
        processed_img = self.clean_ecg_image(image_content)
        
        # Reshape to (1, 224, 224, 3) for model batch requirement
        input_batch = np.expand_dims(processed_img, axis=0)
        
        # Run CNN model inference
        pred_prob = float(self.model.predict(input_batch)[0, 0])
        
        status = "Abnormal ECG (High Risk)" if pred_prob > 0.5 else "Normal ECG (Low Risk)"
        
        return {
            "ecg_risk_probability": pred_prob,
            "ecg_status": status,
            "processed_shape": list(processed_img.shape)
        }
