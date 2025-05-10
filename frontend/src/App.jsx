import React, { useState } from "react";

const App = () => {
  const [formData, setFormData] = useState({
    age: 40,
    hypertension: 0,
    heart_disease: 0,
    avg_glucose_level: 100,
    bmi: 25,
    work_children: false,
    smoke_smokes: false,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let parsedValue = value;
    if (type === "number") {
      parsedValue = parseFloat(value);

      if (name === "age" && (parsedValue < 0 || parsedValue > 120)) return;
      if (name === "avg_glucose_level" && parsedValue < 0) return;
      if (name === "bmi" && parsedValue < 0) return;
    }
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : parsedValue,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    let apiUrl;
    const defaultLocalApiBaseUrl = "http://localhost:9000"; // Default if not set in .env for dev

    if (process.env.NODE_ENV === "production") {
      // In PRODUCTION (when deployed on Vercel)
      // We expect VITE_API_BASE_URL to be set in Vercel's Environment Variables
      const prodApiBaseUrl = import.meta.env.VITE_API_BASE_URL;

      if (!prodApiBaseUrl) {
        console.error(
          "CRITICAL ERROR: VITE_API_BASE_URL is not defined in the production environment!"
        );
        setError(
          "The application is not configured correctly. Please contact support. (Error: API_URL_MISSING)"
        );
        setLoading(false);
        return; // Stop execution if the production API URL is not configured
      }
      apiUrl = `${prodApiBaseUrl}/api/predict`;
    } else {
      // In DEVELOPMENT (when you run npm run dev)
      // Vite will look for VITE_API_BASE_URL_DEV in your frontend/.env.development or frontend/.env.local file
      // If not found, it falls back to the defaultLocalApiBaseUrl
      const devApiBaseUrl =
        import.meta.env.VITE_API_BASE_URL_DEV || defaultLocalApiBaseUrl;
      apiUrl = `${devApiBaseUrl}/api/predict`;
    }

    console.log("Attempting to fetch from:", apiUrl);
    console.log("Form Data:", formData); // Ensure formData is accessible from your component's state

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData), // Ensure formData is correctly stringified
      });

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (parseError) {
          const textError = await response.text();
          console.error(
            "Error response (not JSON or parse failed):",
            textError
          );
          throw new Error(
            textError || `HTTP error! status: ${response.status}`
          );
        }
        console.error("Error response (JSON):", errorData);
        throw new Error(
          errorData.detail ||
            `Prediction failed: ${response.statusText || response.status}`
        );
      }

      const data = await response.json();
      console.log("Prediction Data:", data);
      setPrediction(data);
    } catch (err) {
      console.error("Error in handleSubmit:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getBgColor = (risk) => {
    if (!risk) return "bg-white";

    switch (risk.toLowerCase()) {
      case "low":
        return "bg-green-50 border-l-4 border-green-500";
      case "moderate":
        return "bg-yellow-50 border-l-4 border-yellow-500";
      case "high":
        return "bg-red-50 border-l-4 border-red-500";
      default:
        return "bg-white";
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 font-sans">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Stroke Risk Prediction
        </h1>
        <p className="text-gray-600">
          Enter your health information to assess your stroke risk
        </p>
      </div>

      <div>
        <div className="bg-gray-100 p-6 rounded-lg shadow-md">
          {error && (
            <div
              className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4"
              role="alert"
            >
              <strong className="font-bold">Error!</strong>
              <span className="block sm:inline">{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block font-semibold mb-1 text-gray-700">
                Age:
              </label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-base focus:outline-none focus:ring-2 focus:ring-green-500"
                min="0"
                max="120"
                required
              />
              <small className="block text-gray-500 text-sm mt-1">
                Enter your age in years.
              </small>
            </div>

            <div className="mb-4">
              <label className="block font-semibold mb-1 text-gray-700">
                Hypertension:
              </label>
              <select
                name="hypertension"
                value={formData.hypertension}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-base focus:outline-none focus:ring-2 focus:ring-green-500"
                required
              >
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block font-semibold mb-1 text-gray-700">
                Heart Disease:
              </label>
              <select
                name="heart_disease"
                value={formData.heart_disease}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-base focus:outline-none focus:ring-2 focus:ring-green-500"
                required
              >
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block font-semibold mb-1 text-gray-700">
                Average Glucose Level:
              </label>
              <input
                type="number"
                step="0.01"
                name="avg_glucose_level"
                value={formData.avg_glucose_level}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-base focus:outline-none focus:ring-2 focus:ring-green-500"
                min="0"
                required
              />
              <small className="block text-gray-500 text-sm mt-1">
                Your average blood glucose level (mg/dL).
              </small>
            </div>

            <div className="mb-4">
              <label className="block font-semibold mb-1 text-gray-700">
                BMI:
              </label>
              <input
                type="number"
                step="0.01"
                name="bmi"
                value={formData.bmi}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-base focus:outline-none focus:ring-2 focus:ring-green-500"
                min="0"
                required
              />
              <small className="block text-gray-500 text-sm mt-1">
                Your Body Mass Index.
              </small>
            </div>

            <div className="mb-4">
              <label className="flex items-center font-normal text-gray-700">
                <input
                  type="checkbox"
                  name="work_children"
                  checked={formData.work_children}
                  onChange={handleChange}
                  className="mr-2 h-4 w-4 text-green-500 focus:ring-green-400"
                />
                Work: Children
              </label>
              <small className="block text-gray-500 text-sm mt-1">
                Check if your work type is "children".
              </small>
            </div>

            <div className="mb-6">
              <label className="flex items-center font-normal text-gray-700">
                <input
                  type="checkbox"
                  name="smoke_smokes"
                  checked={formData.smoke_smokes}
                  onChange={handleChange}
                  className="mr-2 h-4 w-4 text-green-500 focus:ring-green-400"
                />
                Currently Smoking
              </label>
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full py-2 px-4 rounded-md text-white font-medium ${
                loading
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-green-500 hover:bg-green-600"
              }`}
            >
              {loading ? "Predicting..." : "Predict Stroke Risk"}
            </button>
          </form>
        </div>

        {prediction && (
          <div
            className={`mt-8 p-6 rounded-lg shadow-md ${getBgColor(
              prediction.stroke_risk
            )}`}
          >
            <h2 className="text-xl font-bold mb-4 text-gray-800">
              Prediction Results
            </h2>
            <p className="text-lg mb-4">
              Risk Level:{" "}
              <strong className="font-bold">{prediction.stroke_risk}</strong>
            </p>
            <div className="bg-white bg-opacity-50 p-4 rounded-md mb-4">
              <p className="mb-2">
                Random Forest: {(prediction.rf_prediction * 100).toFixed(2)}%
              </p>
              <p className="mb-2">
                Support Vector Machine:{" "}
                {prediction.svm_prediction === 0 ? "No Stroke" : "Stroke"}
              </p>
            </div>

            <p className="text-sm text-gray-500 italic">
              Note: This is a prediction based on machine learning models and
              should not replace professional medical advice.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
