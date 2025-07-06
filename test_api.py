#!/usr/bin/env python3
"""
Test script for the Hypertension Prediction API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved:")
            print(f"   Model type: {data['model_type']}")
            print(f"   Features: {data['num_features']}")
            print(f"   Target classes: {data['target_classes']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction...")
    
    # Sample data from the notebook
    test_data = {
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
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Single prediction successful:")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Probability: {data['probability']:.3f}")
            print(f"   Confidence: {data['confidence']}")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    # Sample batch data
    batch_data = [
        {
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
        },
        {
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
        },
        {
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
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful:")
            print(f"   Total patients: {data['total_patients']}")
            for pred in data['predictions']:
                print(f"   Patient {pred['patient_id']}: "
                      f"Prediction={pred['prediction']}, "
                      f"Probability={pred['probability']:.3f}, "
                      f"Confidence={pred['confidence']}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_invalid_input():
    """Test invalid input handling"""
    print("\nTesting invalid input handling...")
    
    # Test with missing required field
    invalid_data = {
        "age": 65.0,
        "sex": 1.0,
        # Missing cp field
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
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        if response.status_code == 422:  # Validation error
            print("✅ Invalid input properly rejected")
            return True
        else:
            print(f"❌ Invalid input not properly handled: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Invalid input test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API tests...")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
        test_invalid_input
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the API implementation.")

if __name__ == "__main__":
    main() 