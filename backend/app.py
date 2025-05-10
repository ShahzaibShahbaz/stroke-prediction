from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv 
load_dotenv()

app = FastAPI(title="Stroke Prediction API")

allowed_origins_str = os.getenv("ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins_list = [origin.strip() for origin in allowed_origins_str.split(',')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list, # <--- USE THE VARIABLE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



SELECTED_FEATURES = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                     'bmi', 'work_children', 'smoke_smokes']


try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models')
    rf_model = joblib.load(os.path.join(model_path, 'stroke_rf.pkl'))
    svm_model = joblib.load(os.path.join(model_path, 'stroke_svm.pkl'))

    
    scaling_params = joblib.load(os.path.join(model_path, 'scaling_params.pkl')) 
    age_mean = scaling_params['age_mean']
    age_std = scaling_params['age_std']
    glucose_mean = scaling_params['glucose_mean']
    glucose_std = scaling_params['glucose_std']
    bmi_mean = scaling_params['bmi_mean']
    bmi_std = scaling_params['bmi_std']

except Exception as e:
    print(f"Error loading models/scaling: {e}")
    models_loaded = False
else:
    models_loaded = True


class StrokeData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    work_children: bool
    smoke_smokes: bool


class PredictionResponse(BaseModel):
    rf_prediction: float
    svm_prediction: int  # SVC gives class (0 or 1), not probability
    stroke_risk: str


@app.post("/predict", response_model=PredictionResponse)
def predict_stroke(data: StrokeData):
    print("Received prediction request...")
    print(f"📊 Data: {data}")
    if not models_loaded:
        raise HTTPException(status_code=503,
                            detail="Machine learning models failed to load. Please check server logs.")
    try:
      
        feature_values = [
            (data.age - age_mean) / age_std,
            data.hypertension,
            data.heart_disease,
            (data.avg_glucose_level - glucose_mean) / glucose_std,
            (data.bmi - bmi_mean) / bmi_std,
            1 if data.work_children else 0,
            1 if data.smoke_smokes else 0
        ]

      
        features = np.array(feature_values).reshape(1, -1)

        print("Feature Shape:", features.shape)
        print("Feature Data:", features)

        rf_pred = rf_model.predict_proba(features)[0][1]  
        svm_pred = svm_model.predict(features)[0]  

        # Adjust the averaging (this is the key change!)
        avg_prediction = (rf_pred + svm_pred) / 2

        if avg_prediction < 0.2:
            risk = "Low"
        elif avg_prediction < 0.5:
            risk = "Moderate"
        else:
            risk = "High"

        print(f"Prediction complete: RF={rf_pred:.2f}, SVM={svm_pred}, Risk={risk}")

        return {
            "rf_prediction": float(rf_pred),
            "svm_prediction": int(svm_pred),  # Return int for SVC
            "stroke_risk": risk
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def read_root():
    try:
        status = "OK" if models_loaded else "Models Failed to Load"
        return {
            "status": status,
        }
    except Exception as e:
        print(f"Error in root endpoint: {e}")
        return {"error": str(e)}


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": models_loaded}


@app.on_event("startup")
async def startup_event():
    print("FastAPI App is fully initialized and ready to handle requests.")


if __name__ == "__main__":
   
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 9000)) # os.getenv returns string, so convert port to int

    print(f"🔧 Uvicorn server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)