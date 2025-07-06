from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the request body structure with validation
class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age of the patient")
    sex: float = Field(..., ge=0, le=1, description="Sex (0: Female, 1: Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0: No, 1: Yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., ge=0, le=300, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0: No, 1: Yes)")
    oldpeak: float = Field(..., ge=-10, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0: No hypertension, 1: Hypertension)")
    probability: float = Field(..., ge=0, le=1, description="Probability of hypertension")
    confidence: str = Field(..., description="Confidence level based on probability")

# Initialize FastAPI app
app = FastAPI(
    title="Hypertension Prediction API",
    description="A machine learning API for predicting hypertension based on patient features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variable to store the model
model_pipeline = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model_pipeline
    try:
        model_pipeline = joblib.load("logistic_pipeline_model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hypertension Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_hypertension(data: PredictionRequest):
    """
    Predict hypertension based on patient features
    
    Returns:
    - prediction: 0 (no hypertension) or 1 (hypertension)
    - probability: Probability of hypertension (0-1)
    - confidence: Confidence level (Low/Medium/High)
    """
    try:
        if model_pipeline is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert request data to DataFrame
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[:, 1][0]
        
        # Determine confidence level
        if probability < 0.3 or probability > 0.7:
            confidence = "High"
        elif probability < 0.4 or probability > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        logger.info(f"Prediction made: {prediction}, Probability: {probability:.3f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get feature names from the pipeline
        feature_names = model_pipeline.named_steps['scaler'].feature_names_in_
        
        return {
            "model_type": "Logistic Regression Pipeline",
            "features": feature_names.tolist(),
            "num_features": len(feature_names),
            "target_classes": [0, 1],
            "target_description": {
                0: "No hypertension",
                1: "Hypertension"
            }
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")

@app.post("/predict-batch")
async def predict_batch(data_list: list[PredictionRequest]):
    """
    Predict hypertension for multiple patients at once
    
    Returns predictions for all patients in the batch
    """
    try:
        if model_pipeline is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if len(data_list) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 predictions allowed.")
        
        # Convert request data to DataFrame
        input_data_list = [data.model_dump() for data in data_list]
        input_df = pd.DataFrame(input_data_list)
        
        # Make predictions
        predictions = model_pipeline.predict(input_df)
        probabilities = model_pipeline.predict_proba(input_df)[:, 1]
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if prob < 0.3 or prob > 0.7:
                confidence = "High"
            elif prob < 0.4 or prob > 0.6:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            results.append({
                "patient_id": i,
                "prediction": int(pred),
                "probability": float(prob),
                "confidence": confidence
            })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        
        return {
            "predictions": results,
            "total_patients": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 