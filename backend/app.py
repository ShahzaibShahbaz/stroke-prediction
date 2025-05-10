from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stroke Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models
lr_model = joblib.load('stroke_lr.pkl')
rf_model = joblib.load('stroke_rf.pkl')
svm_model = joblib.load('stroke_svm.pkl')

class StrokeData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    work_children: bool
    smoke_smokes: bool

class PredictionResponse(BaseModel):
    lr_prediction: float
    rf_prediction: float
    svm_prediction: float
    stroke_risk: str

@app.post("/predict", response_model=PredictionResponse)
def predict_stroke(data: StrokeData):
    try:
        # Prepare features
        features = np.array([[
            data.age,
            data.hypertension,
            data.heart_disease,
            data.avg_glucose_level,
            data.bmi,
            data.work_children,
            data.smoke_smokes
        ]])
        
        # Get predictions from each model
        lr_pred = lr_model.predict_proba(features)[0][1]
        rf_pred = rf_model.predict_proba(features)[0][1]
        svm_pred = svm_model.predict_proba(features)[0][1]
        
        # Simple ensemble (average)
        avg_prediction = (lr_pred + rf_pred + svm_pred) / 3
        
        # Determine risk level
        if avg_prediction < 0.2:
            risk = "Low"
        elif avg_prediction < 0.5:
            risk = "Moderate"
        else:
            risk = "High"
        
        return {
            "lr_prediction": float(lr_pred),
            "rf_prediction": float(rf_pred),
            "svm_prediction": float(svm_pred),
            "stroke_risk": risk
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Stroke Prediction API is running. Use POST /predict endpoint to get predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)