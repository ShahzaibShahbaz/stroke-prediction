# stroke-prediction-app/api/index.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

# --- Environment Variable and Path Configuration ---
# Attempt to load .env for local development.
# On Vercel, environment variables are set in the dashboard.
IS_VERCEL_ENV = os.getenv("VERCEL_ENV") is not None # Vercel sets VERCEL_ENV (e.g., "production", "preview", "development")

try:
    from dotenv import load_dotenv
    if not IS_VERCEL_ENV: # Only load .env if not on Vercel
        # For local: assumes .env is in the project root (stroke-prediction-app/.env)
        # and this script (index.py) is in stroke-prediction-app/api/
        project_root_for_dotenv = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dotenv_path = os.path.join(project_root_for_dotenv, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            print(f"LOCAL: Loaded .env file from: {dotenv_path}")
        else:
            print(f"LOCAL: .env file not found at {dotenv_path}. Relying on system env vars.")
    else:
        print("VERCEL ENV: Skipping .env load, relying on Vercel environment variables.")
except ImportError:
    print("python-dotenv not installed. Relying on system environment variables.")

app = FastAPI(title="Stroke Prediction API")

# --- CORS Configuration ---
default_origins = "http://localhost:5173,http://127.0.0.1:5173"

allowed_origins_str = os.getenv("ALLOW_ORIGINS", default_origins)
allowed_origins_list = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]
print(f"CORS: Allowed Origins List: {allowed_origins_list}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Path Configuration ---
if IS_VERCEL_ENV:
    # On Vercel, this script (index.py) is in /var/task/api/ (or similar)
    # The 'models' directory will be at /var/task/models/
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../api/
    project_root_dir = os.path.dirname(current_script_dir)        # .../
    MODEL_DIR = os.path.join(project_root_dir, 'models')
    print(f"VERCEL ENV: Model directory configured to: {MODEL_DIR}")
else:
    # LOCAL: Your original logic was base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # If index.py is in 'stroke-prediction-app/api/', then:
    # os.path.abspath(__file__) is '.../stroke-prediction-app/api/index.py'
    # os.path.dirname(os.path.abspath(__file__)) is '.../stroke-prediction-app/api/'
    # os.path.dirname(os.path.dirname(os.path.abspath(__file__))) IS '.../stroke-prediction-app/'
    # So, MODEL_DIR should be '.../stroke-prediction-app/models/'
    base_dir_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(base_dir_local, 'models')
    print(f"LOCAL ENV: Model directory configured to: {MODEL_DIR}")

# --- Global variables for models and parameters ---
models_loaded = False
rf_model = None
svm_model = None
# Use a distinct name for the dictionary to avoid confusion with the filename
scaling_params_data = {}
age_mean, age_std, glucose_mean, glucose_std, bmi_mean, bmi_std = [None] * 6

# --- Model Loading Logic ---
try:
    rf_model_path = os.path.join(MODEL_DIR, 'stroke_rf.pkl')
    svm_model_path = os.path.join(MODEL_DIR, 'stroke_svm.pkl')
    scaling_params_path = os.path.join(MODEL_DIR, 'scaling_params.pkl')

    print(f"Attempting to load models from: {MODEL_DIR}")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Fatal: Models directory itself not found at {MODEL_DIR}")

    print(f"Loading RF model: {rf_model_path}")
    if not os.path.exists(rf_model_path): raise FileNotFoundError(f"RF model not found at {rf_model_path}")
    rf_model = joblib.load(rf_model_path)
    print("RF model loaded.")

    print(f"Loading SVM model: {svm_model_path}")
    if not os.path.exists(svm_model_path): raise FileNotFoundError(f"SVM model not found at {svm_model_path}")
    svm_model = joblib.load(svm_model_path)
    print("SVM model loaded.")

    print(f"Loading scaling params: {scaling_params_path}")
    if not os.path.exists(scaling_params_path): raise FileNotFoundError(f"Scaling params not found at {scaling_params_path}")
    scaling_params_data = joblib.load(scaling_params_path)
    print(f"Scaling params loaded. Keys: {scaling_params_data.keys()}")

    age_mean = scaling_params_data['age_mean']
    age_std = scaling_params_data['age_std']
    glucose_mean = scaling_params_data['glucose_mean']
    glucose_std = scaling_params_data['glucose_std']
    bmi_mean = scaling_params_data['bmi_mean']
    bmi_std = scaling_params_data['bmi_std']

    if not all([p is not None for p in [age_mean, age_std, glucose_mean, glucose_std, bmi_mean, bmi_std]]):
        raise ValueError("One or more scaling parameters are None after attempting to load.")

    models_loaded = True
    print("All models and scaling parameters processed successfully.")

except FileNotFoundError as fnf_error:
    print(f"MODEL LOADING ERROR (FileNotFound): {fnf_error}")
except KeyError as ke_error:
    print(f"MODEL LOADING ERROR (KeyError in scaling_params): {ke_error}")
except ValueError as ve_error:
    print(f"MODEL LOADING ERROR (ValueError regarding scaling_params): {ve_error}")
except Exception as e:
    print(f"GENERAL MODEL LOADING ERROR: {e}")
    import traceback
    traceback.print_exc()
# --- End Model Loading ---

# ... (rest of your Pydantic models and endpoint definitions: StrokeData, PredictionResponse, /predict, /, /health)
# Ensure they are identical to your last provided backend code, minus the uvicorn and startup event.
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
    svm_prediction: int
    stroke_risk: str


@app.post("/api/predict", response_model=PredictionResponse) # On Vercel, this is /api/predict
def predict_stroke(data: StrokeData):
    print(f"Request to /predict. Data: {data.model_dump_json()}")
    if not models_loaded:
        print("Error in /predict: Models not loaded.")
        raise HTTPException(status_code=503,
                            detail="Machine learning models are currently unavailable.")
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

        rf_pred_prob = rf_model.predict_proba(features)[0][1]
        svm_pred_class = int(svm_model.predict(features)[0]) # Ensure it's a Python int

        # Your averaging logic
        avg_prediction = (rf_pred_prob + float(svm_pred_class)) / 2

        if avg_prediction < 0.2:
            risk = "Low"
        elif avg_prediction < 0.5:
            risk = "Moderate"
        else:
            risk = "High"

        print(f"Prediction successful: RF={rf_pred_prob:.4f}, SVM={svm_pred_class}, Avg={avg_prediction:.4f}, Risk={risk}")
        return {
            "rf_prediction": float(rf_pred_prob),
            "svm_prediction": svm_pred_class,
            "stroke_risk": risk
        }
    except Exception as e:
        print(f"Error during prediction in /predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error.")

@app.get("/") # On Vercel, this is /api/
def read_root():
    return {
        "message": "Stroke Prediction API (Vercel)",
        "models_loaded": models_loaded,
        "model_directory_used": MODEL_DIR,
        "is_vercel_environment": IS_VERCEL_ENV,
        "expected_scaling_param_keys": ['age_mean', 'age_std', 'glucose_mean', 'glucose_std', 'bmi_mean', 'bmi_std'],
        "loaded_scaling_param_keys": list(scaling_params_data.keys()) if scaling_params_data else "None"
    }

@app.get("/health") # On Vercel, this is /api/health
def health_check():
    all_params_loaded = all(p is not None for p in [age_mean, age_std, glucose_mean, glucose_std, bmi_mean, bmi_std])
    return {
        "status": "healthy" if models_loaded and all_params_loaded else "degraded",
        "models_loaded_flag": models_loaded,
        "rf_model_ok": rf_model is not None,
        "svm_model_ok": svm_model is not None,
        "scaling_params_ok": all_params_loaded
        }

