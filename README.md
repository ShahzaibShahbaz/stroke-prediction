# Stroke Risk Prediction Application

This is a full-stack web application designed to predict the risk of stroke based on user-provided health information. It utilizes a React front-end for user interaction and a Python FastAPI back-end to serve machine learning model predictions.

## Features

- **User-friendly Interface:** Simple form to input health metrics.
- **Dual Model Prediction:**
  - **Random Forest (RF):** Provides a probability score for stroke risk.
  - **Support Vector Machine (SVM):** Provides a class prediction (Stroke/No Stroke).
- **Risk Categorization:** Combines model outputs to categorize risk as "Low," "Moderate," or "High."
- **Real-time Feedback:** Displays prediction results dynamically.
- **Responsive Design:** Basic styling suitable for various screen sizes (due to Tailwind CSS).
- **CORS Enabled:** Allows the front-end (running on `http://localhost:5173`) to communicate with the back-end API.

## Technologies Used

**Front-end:**

- React (with Hooks)
- JavaScript (ES6+)
- Tailwind CSS (for styling)
- Fetch API (for HTTP requests)

**Back-end:**

- Python 3.x
- FastAPI (for building the API)
- Uvicorn (ASGI server)
- Joblib (for loading pre-trained ML models)
- NumPy (for numerical operations)
- Pydantic (for data validation)

**Machine Learning Models (Pre-trained):**

- Random Forest Classifier (`stroke_rf.pkl`)
- Support Vector Classifier (`stroke_svm.pkl`)
- Scaling Parameters (`scaling_params.pkl` for standardizing numerical features)

## Project Structure (Recommended)

```
stroke-prediction-app/
├── backend/
│   ├── app.py                # FastAPI application
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── public/
│   ├── src/
│   │   └── App.js            # Main React component
│   ├── package.json          # Node dependencies
│   └── ... (other React files)
├── models/                   # ML models and scaling parameters
│   ├── stroke_rf.pkl
│   ├── stroke_svm.pkl
│   └── scaling_params.pkl
└── README.md                 # This file
```

## Prerequisites

- Node.js and npm (or yarn) for the front-end.
- Python 3.7+ and pip for the back-end.
- The pre-trained model files (`stroke_rf.pkl`, `stroke_svm.pkl`) and `scaling_params.pkl` must be available.

## Setup and Installation

**1. Clone the Repository (if applicable):**

```bash
git clone <your-repository-url>
cd stroke-prediction-app
```

**2. Backend Setup:**

a. **Navigate to the backend directory:**

```bash
cd backend
```

b. **Create and place model files:**
Ensure you have a `models` directory **one level above** the `backend` directory (i.e., in the project root `stroke-prediction-app/models/`) and place the following files inside it:

- `stroke_rf.pkl`
- `stroke_svm.pkl`
- `scaling_params.pkl`

_The backend script `app.py` expects the `models` directory at `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` which resolves to the parent directory of the `backend` directory._

c. **Create a Python virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

d. **Create `requirements.txt`:**
Create a file named `requirements.txt` in the `backend` directory with the following content (derived from your provided list):

```
fastapi==0.115.12
uvicorn[standard]==0.34.2 # [standard] includes websockets and httptools
joblib==1.5.0
numpy==2.2.5
pydantic==2.11.4
python-dotenv==1.1.0 # If you plan to use .env files
# Add other direct dependencies if any were missed from the list
# For example, if your models require scikit-learn to be explicitly installed:
# scikit-learn
```

e. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**3. Frontend Setup:**

a. **Navigate to the frontend directory:**

```bash
cd ../frontend  # Or cd path/to/your/frontend
```

b. **Install Node.js dependencies:**

```bash
npm install
# or
# yarn install
```

## Running the Application

**1. Start the Backend Server:**

- Ensure you are in the `backend` directory.
- Activate the virtual environment if not already active.
- Run the FastAPI application using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 9000 --reload
```

The backend API will be available at `http://localhost:9000`. The `--reload` flag enables auto-reloading on code changes (useful for development).

**2. Start the Frontend Development Server:**

- Ensure you are in the `frontend` directory.
- Run the React development server:

```bash
npm start
# or
# yarn start
```

The front-end application will typically open automatically in your browser at `http://localhost:5173` (as specified in your backend CORS settings and is common for Vite-based React apps) or `http://localhost:3000` (common for Create React App).

**3. Access the Application:**
Open your web browser and navigate to `http://localhost:5173` (or the port your frontend is running on).

## How It Works

1. The user inputs their health data into the form on the React front-end.
2. Upon submission, the front-end sends a `POST` request with the data in JSON format to the `/predict` endpoint of the FastAPI back-end (`http://localhost:9000/predict`).
3. The FastAPI backend:
   - Receives the data and validates it using the `StrokeData` Pydantic model.
   - **Preprocesses the input:**
     - Converts boolean inputs (`work_children`, `smoke_smokes`) to integers (0 or 1).
     - Scales `age`, `avg_glucose_level`, and `bmi` using pre-calculated mean and standard deviation values loaded from `scaling_params.pkl`.
   - Feeds the processed features into the pre-trained Random Forest (`stroke_rf.pkl`) and SVM (`stroke_svm.pkl`) models.
   - The RF model outputs a probability for class 1 (stroke).
   - The SVM model outputs a class prediction (0 for no stroke, 1 for stroke).
   - It calculates an `avg_prediction` by `(rf_pred_probability + svm_pred_class) / 2`.
   - Based on this `avg_prediction`, it determines a risk level:
     - `Low`: if `avg_prediction < 0.2`
     - `Moderate`: if `avg_prediction < 0.5`
     - `High`: otherwise
   - Returns the RF probability, SVM class prediction, and the determined `stroke_risk` as a JSON response.
4. The React front-end receives the response and updates the UI to display the prediction results, including the risk level and individual model outputs, with color-coding for the risk.

## API Endpoints (Backend)

- **`POST /predict`**:
  - **Request Body:** JSON object with user health data.
    ```json
    {
      "age": 40,
      "hypertension": 0, // 0 for No, 1 for Yes
      "heart_disease": 0, // 0 for No, 1 for Yes
      "avg_glucose_level": 100.0,
      "bmi": 25.0,
      "work_children": false, // true or false
      "smoke_smokes": false // true or false
    }
    ```
  - **Response Body (Success - 200 OK):**
    ```json
    {
      "rf_prediction": 0.15, // Example RF probability
      "svm_prediction": 0, // Example SVM class (0 or 1)
      "stroke_risk": "Low" // "Low", "Moderate", or "High"
    }
    ```
  - **Response Body (Error - 503 Service Unavailable):** If models fail to load.
    ```json
    {
      "detail": "Machine learning models failed to load. Please check server logs."
    }
    ```
  - **Response Body (Error - 500 Internal Server Error):** For other prediction errors.
    ```json
    { "detail": "Prediction failed: <error_message>" }
    ```
- **`GET /`**:
  - **Response Body:** Basic API status.
    ```json
    { "status": "OK" } // or "Models Failed to Load"
    ```
- **`GET /health`**:
  - **Response Body:** Health check indicating if models are loaded.
    ```json
    { "status": "healthy", "models_loaded": true } // or false
    ```

## Input Features

The models expect the following features:

1. `age`: (Float) Age of the person.
2. `hypertension`: (Integer) 0 if no hypertension, 1 if has hypertension.
3. `heart_disease`: (Integer) 0 if no heart disease, 1 if has heart disease.
4. `avg_glucose_level`: (Float) Average glucose level in blood.
5. `bmi`: (Float) Body Mass Index.
6. `work_children`: (Integer) 1 if work type is "children", 0 otherwise.
7. `smoke_smokes`: (Integer) 1 if currently smokes, 0 otherwise.

## Important Note

This application is for informational and educational purposes only. The predictions made by the machine learning models should **not** be considered as medical advice. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health.
