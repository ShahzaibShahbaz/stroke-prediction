# stroke-prediction-app/api/index.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

# --- Initial Debug Prints (Add these at the very top) ---
print("--- START OF API/INDEX.PY EXECUTION ---")
print(f"Current working directory: {os.getcwd()}")
print(f"Absolute path of __file__: {os.path.abspath(__file__)}")

# --- Environment Variable and Path Configuration ---
IS_VERCEL_ENV = os.getenv("VERCEL_ENV") is not None
print(f"IS_VERCEL_ENV: {IS_VERCEL_ENV}")

try:
    from dotenv import load_dotenv
    if not IS_VERCEL_ENV:
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
print(f"CORS: Allowed Origins List: {allowed_origins_list}") # This was printing before

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware added successfully.") # New print

# --- Model Path Configuration (Enhanced Logging) ---
print("Starting model path configuration...")
MODEL_DIR = None
VERCEL_EXPECTED_PROJECT_ROOT = "/var/task" # Standard Vercel root for Python functions

if IS_VERCEL_ENV:
    print(f"VERCEL ENV: Detected. __file__ is: {__file__}")
    # Vercel's includeFiles places items from project root into the deployment's root.
    # If 'models' is at your project root, it should appear in VERCEL_EXPECTED_PROJECT_ROOT.
    MODEL_DIR = os.path.join(VERCEL_EXPECTED_PROJECT_ROOT, "models")
    print(f"VERCEL ENV: Tentative MODEL_DIR based on Vercel structure: {MODEL_DIR}")

    # For verification, let's check the original logic's derived path
    try:
        abs_file_path_check = os.path.abspath(__file__)
        current_script_dir_check = os.path.dirname(abs_file_path_check)
        project_root_dir_check = os.path.dirname(current_script_dir_check)
        derived_model_dir_check = os.path.join(project_root_dir_check, 'models')
        print(f"VERCEL ENV (Path Check): os.path.abspath(__file__) = {abs_file_path_check}")
        print(f"VERCEL ENV (Path Check): current_script_dir_check = {current_script_dir_check}")
        print(f"VERCEL ENV (Path Check): project_root_dir_check = {project_root_dir_check}")
        print(f"VERCEL ENV (Path Check): derived_model_dir_check = {derived_model_dir_check}")
        if MODEL_DIR != derived_model_dir_check:
            print(f"WARNING: Tentative MODEL_DIR ({MODEL_DIR}) differs from derived_model_dir_check ({derived_model_dir_check}). Sticking with tentative.")
    except Exception as e_path:
        print(f"VERCEL ENV: Error during diagnostic path derivation: {e_path}")
else:
    print("LOCAL ENV: Detected.")
    # This assumes api/index.py is in an 'api' subdirectory
    # and 'models' is at the root of your project.
    base_dir_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(base_dir_local, 'models')
    print(f"LOCAL ENV: Model directory configured to: {MODEL_DIR}")

if MODEL_DIR is None:
    print("CRITICAL ERROR: MODEL_DIR was not set. Halting before model load attempt.")
    # You might want to raise an exception here or ensure models_loaded remains False
else:
    print(f"FINAL MODEL_DIR selected for use: {MODEL_DIR}")

# --- Global variables for models and parameters ---
models_loaded = False
rf_model = None
svm_model = None
scaling_params_data = {}
age_mean, age_std, glucose_mean, glucose_std, bmi_mean, bmi_std = [None] * 6

# --- Model Loading Logic (Enhanced Logging) ---
# This check is crucial before the try-except block for model loading
if MODEL_DIR: # Proceed only if MODEL_DIR is set
    try:
        print(f"Attempting to load models from: {MODEL_DIR}")
        if not os.path.exists(MODEL_DIR):
            print(f"FATAL: Models directory itself NOT FOUND at {MODEL_DIR}")
            # To understand why, let's see what's in the expected parent directory on Vercel
            if IS_VERCEL_ENV:
                print(f"Listing contents of VERCEL_EXPECTED_PROJECT_ROOT ({VERCEL_EXPECTED_PROJECT_ROOT}):")
                try:
                    print(os.listdir(VERCEL_EXPECTED_PROJECT_ROOT))
                except Exception as e_ls:
                    print(f"Could not list contents of {VERCEL_EXPECTED_PROJECT_ROOT}: {e_ls}")
            raise FileNotFoundError(f"Models directory not found at {MODEL_DIR}") # Explicitly raise

        print(f"Confirmed: Models directory exists at {MODEL_DIR}. Contents: {os.listdir(MODEL_DIR)}")

        rf_model_path = os.path.join(MODEL_DIR, 'stroke_rf.pkl')
        svm_model_path = os.path.join(MODEL_DIR, 'stroke_svm.pkl')
        scaling_params_path = os.path.join(MODEL_DIR, 'scaling_params.pkl')

        print(f"Loading RF model from: {rf_model_path}. Exists: {os.path.exists(rf_model_path)}")
        if not os.path.exists(rf_model_path): raise FileNotFoundError(f"RF model not found at {rf_model_path}")
        rf_model = joblib.load(rf_model_path)
        print("RF model loaded successfully.")

        print(f"Loading SVM model from: {svm_model_path}. Exists: {os.path.exists(svm_model_path)}")
        if not os.path.exists(svm_model_path): raise FileNotFoundError(f"SVM model not found at {svm_model_path}")
        svm_model = joblib.load(svm_model_path)
        print("SVM model loaded successfully.")

        print(f"Loading scaling params from: {scaling_params_path}. Exists: {os.path.exists(scaling_params_path)}")
        if not os.path.exists(scaling_params_path): raise FileNotFoundError(f"Scaling params not found at {scaling_params_path}")
        scaling_params_data = joblib.load(scaling_params_path)
        print(f"Scaling params loaded. Keys: {list(scaling_params_data.keys())}")

        age_mean = scaling_params_data['age_mean']
        age_std = scaling_params_data['age_std']
        glucose_mean = scaling_params_data['glucose_mean']
        glucose_std = scaling_params_data['glucose_std']
        bmi_mean = scaling_params_data['bmi_mean']
        bmi_std = scaling_params_data['bmi_std']
        print("Scaling parameters extracted successfully.")

        if not all([p is not None for p in [age_mean, age_std, glucose_mean, glucose_std, bmi_mean, bmi_std]]):
            print("ERROR: One or more scaling parameters are None after loading.")
            raise ValueError("One or more scaling parameters are None after attempting to load.")

        models_loaded = True
        print("All models and scaling parameters processed successfully. models_loaded = True")

    except FileNotFoundError as fnf_error:
        print(f"MODEL LOADING ERROR (FileNotFound): {fnf_error}")
        models_loaded = False # Ensure it's false on error
    except KeyError as ke_error:
        print(f"MODEL LOADING ERROR (KeyError in scaling_params): {ke_error}")
        models_loaded = False # Ensure it's false on error
    except ValueError as ve_error:
        print(f"MODEL LOADING ERROR (ValueError regarding scaling_params): {ve_error}")
        models_loaded = False # Ensure it's false on error
    except Exception as e:
        print(f"GENERAL MODEL LOADING ERROR: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc() # This will print the full stack trace to Vercel logs
        models_loaded = False # Ensure it's false on error
else:
    print("CRITICAL: MODEL_DIR was None, so model loading was skipped.")
    models_loaded = False
# --- End Model Loading ---

# ... (rest of your Pydantic models and endpoint definitions: StrokeData, PredictionResponse, /predict, /, /health) ...
# Ensure StrokeData, PredictionResponse, predict_stroke, read_root, health_check are identical to your original.
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
        print("Error in /predict: Models not loaded. models_loaded is False.") # Clarify state
        # Adding current MODEL_DIR state for context in case of failure
        print(f"Context: MODEL_DIR was '{MODEL_DIR}'. Check earlier logs for loading issues.")
        raise HTTPException(status_code=503,
                            detail="Machine learning models are currently unavailable. Please check server logs.")
    try:
        # ... (your prediction logic remains the same)
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
        svm_pred_class = int(svm_model.predict(features)[0])

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
        print(f"Error during prediction in /predict: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")

@app.get("/") # On Vercel, this is /api/
def read_root():
    return {
        "message": "Stroke Prediction API",
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
        "scaling_params_ok": all_params_loaded,
        "model_directory_checked": MODEL_DIR
    }