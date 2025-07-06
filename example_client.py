#!/usr/bin/env python3
"""
Example client for the Hypertension Prediction API
"""

import requests
import json
import time

class HypertensionAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model-info")
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"Model info request failed: {e}")
            return None
    
    def predict_single(self, patient_data):
        """Predict hypertension for a single patient"""
        try:
            response = requests.post(f"{self.base_url}/predict", json=patient_data)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"Single prediction failed: {e}")
            return None
    
    def predict_batch(self, patients_data):
        """Predict hypertension for multiple patients"""
        try:
            response = requests.post(f"{self.base_url}/predict-batch", json=patients_data)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"Batch prediction failed: {e}")
            return None

def main():
    """Example usage of the API client"""
    
    # Initialize client
    client = HypertensionAPIClient()
    
    print("🏥 Hypertension Prediction API Client")
    print("=" * 50)
    
    # Check API health
    print("1. Checking API health...")
    health = client.health_check()
    if health:
        print(f"✅ API is healthy: {health}")
    else:
        print("❌ API is not responding")
        return
    
    # Get model info
    print("\n2. Getting model information...")
    model_info = client.get_model_info()
    if model_info:
        print(f"✅ Model info retrieved:")
        print(f"   Model type: {model_info['model_type']}")
        print(f"   Features: {model_info['num_features']}")
        print(f"   Target classes: {model_info['target_classes']}")
    else:
        print("❌ Failed to get model info")
        return
    
    # Example patient data
    patient_1 = {
        "age": 65.0,
        "sex": 1.0,
        "cp": 3,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.0,
        "slope": 1,
        "ca": 0,
        "thal": 3
    }
    
    patient_2 = {
        "age": 45.0,
        "sex": 0.0,
        "cp": 1,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    patient_3 = {
        "age": 55.0,
        "sex": 1.0,
        "cp": 2,
        "trestbps": 130,
        "chol": 220,
        "fbs": 1,
        "restecg": 1,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 1,
        "thal": 3
    }
    
    # Single prediction
    print("\n3. Making single prediction...")
    result = client.predict_single(patient_1)
    if result:
        print(f"✅ Single prediction result:")
        print(f"   Prediction: {result['prediction']} ({'Hypertension' if result['prediction'] == 1 else 'No Hypertension'})")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']}")
    else:
        print("❌ Single prediction failed")
    
    # Batch prediction
    print("\n4. Making batch prediction...")
    batch_result = client.predict_batch([patient_1, patient_2, patient_3])
    if batch_result:
        print(f"✅ Batch prediction results:")
        print(f"   Total patients: {batch_result['total_patients']}")
        for pred in batch_result['predictions']:
            status = "Hypertension" if pred['prediction'] == 1 else "No Hypertension"
            print(f"   Patient {pred['patient_id']}: {status} (prob: {pred['probability']:.3f}, conf: {pred['confidence']})")
    else:
        print("❌ Batch prediction failed")
    
    print("\n" + "=" * 50)
    print("🎉 API client demonstration completed!")

if __name__ == "__main__":
    main() 