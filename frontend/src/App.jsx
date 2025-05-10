import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    age: 0,
    hypertension: 0,
    heart_disease: 0,
    avg_glucose_level: 0,
    bmi: 0,
    work_children: false,
    smoke_smokes: false,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]:
        type === "checkbox"
          ? checked
          : type === "number"
          ? parseFloat(value)
          : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while making the prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stroke Risk Prediction</h1>
        <p>Enter your health information to assess your stroke risk</p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-group">
            <label>Age (standardized):</label>
            <input
              type="number"
              step="0.01"
              name="age"
              value={formData.age}
              onChange={handleChange}
              required
            />
            <small>Enter your age, standardized as per the model</small>
          </div>

          <div className="form-group">
            <label>Hypertension:</label>
            <select
              name="hypertension"
              value={formData.hypertension}
              onChange={handleChange}
              required
            >
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div className="form-group">
            <label>Heart Disease:</label>
            <select
              name="heart_disease"
              value={formData.heart_disease}
              onChange={handleChange}
              required
            >
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div className="form-group">
            <label>Average Glucose Level (standardized):</label>
            <input
              type="number"
              step="0.01"
              name="avg_glucose_level"
              value={formData.avg_glucose_level}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>BMI (standardized):</label>
            <input
              type="number"
              step="0.01"
              name="bmi"
              value={formData.bmi}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group checkbox">
            <label>
              <input
                type="checkbox"
                name="work_children"
                checked={formData.work_children}
                onChange={handleChange}
              />
              Work: Children
            </label>
          </div>

          <div className="form-group checkbox">
            <label>
              <input
                type="checkbox"
                name="smoke_smokes"
                checked={formData.smoke_smokes}
                onChange={handleChange}
              />
              Currently Smoking
            </label>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Stroke Risk"}
          </button>
        </form>

        {prediction && (
          <div
            className={`prediction-result ${prediction.stroke_risk.toLowerCase()}`}
          >
            <h2>Prediction Results</h2>
            <p className="risk-level">
              Risk Level: <strong>{prediction.stroke_risk}</strong>
            </p>
            <div className="model-predictions">
              <p>
                Logistic Regression:{" "}
                {(prediction.lr_prediction * 100).toFixed(2)}%
              </p>
              <p>
                Random Forest: {(prediction.rf_prediction * 100).toFixed(2)}%
              </p>
              <p>
                Support Vector Machine:{" "}
                {(prediction.svm_prediction * 100).toFixed(2)}%
              </p>
            </div>
            <p className="disclaimer">
              Note: This is a prediction based on machine learning models and
              should not replace professional medical advice.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
