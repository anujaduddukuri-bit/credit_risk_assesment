from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ======================================================
# Base Directory
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ======================================================
# Model & Encoder Paths
# ======================================================

MODEL_PATH = os.path.join(MODELS_DIR, "extra_trees_credit_model.pkl")
SEX_ENCODER_PATH = os.path.join(MODELS_DIR, "Sex_encoder.pkl")
HOUSING_ENCODER_PATH = os.path.join(MODELS_DIR, "Housing_encoder.pkl")
SAVING_ENCODER_PATH = os.path.join(MODELS_DIR, "Saving accounts_encoder.pkl")
CHECKING_ENCODER_PATH = os.path.join(MODELS_DIR, "Checking account_encoder.pkl")
TARGET_ENCODER_PATH = os.path.join(MODELS_DIR, "target_encoder.pkl")

# ======================================================
# Check Required Files
# ======================================================

required_files = [
    MODEL_PATH,
    SEX_ENCODER_PATH,
    HOUSING_ENCODER_PATH,
    SAVING_ENCODER_PATH,
    CHECKING_ENCODER_PATH,
    TARGET_ENCODER_PATH
]

for file in required_files:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Missing file:\n{file}")

# ======================================================
# Load Model & Encoders
# ======================================================

try:
    model = joblib.load(MODEL_PATH)

    sex_encoder = joblib.load(SEX_ENCODER_PATH)
    housing_encoder = joblib.load(HOUSING_ENCODER_PATH)
    saving_encoder = joblib.load(SAVING_ENCODER_PATH)
    checking_encoder = joblib.load(CHECKING_ENCODER_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)

except Exception as e:
    raise RuntimeError(f"Error loading model or encoders:\n{e}")

# ======================================================
# Purpose Encoding
# ======================================================

purpose_mapping = {
    "car": 0,
    "radio/TV": 1,
    "education": 2,
    "furniture/equipment": 3,
    "business": 4,
    "domestic appliances": 5,
    "repairs": 6,
    "vacation/others": 7
}

# ======================================================
# Home
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")

# ======================================================
# About
# ======================================================

@app.route("/about")
def about():
    return render_template("about.html")

# ======================================================
# Prediction
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():
    print("Predict route called")

    try:

        age = int(request.form["age"])
        sex = request.form["sex"]
        job = int(request.form["job"])
        housing = request.form["housing"]
        saving = request.form["saving"]
        checking = request.form["checking"]
        credit_amount = float(request.form["credit_amount"])
        duration = int(request.form["duration"])
        purpose = request.form["purpose"]

        # Encode categorical values
        sex = sex_encoder.transform([sex])[0]
        housing = housing_encoder.transform([housing])[0]
        saving = saving_encoder.transform([saving])[0]
        checking = checking_encoder.transform([checking])[0]
        purpose = purpose_mapping.get(purpose, 0)

        # Feature Vector
        features = np.array([[
            age,
            sex,
            job,
            housing,
            saving,
            checking,
            credit_amount,
            duration,
            purpose
        ]])

        # Prediction
        prediction = model.predict(features)[0]

        # Probability
        probability = 100

        if hasattr(model, "predict_proba"):
            probability = float(np.max(model.predict_proba(features)) * 100)

        # Decode prediction
        prediction_label = target_encoder.inverse_transform([prediction])[0]

        # Recommendation
        if str(prediction_label).lower() == "good":

            recommendation = (
                "Loan can be approved. Applicant has a good credit profile."
            )

        else:

            recommendation = (
                "Loan approval is not recommended. Applicant has high credit risk."
            )

        return render_template(
            "result.html",
            prediction=prediction_label,
            probability=round(probability, 2),
            recommendation=recommendation
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction="Prediction Failed",
            probability=0,
            recommendation=str(e)
        )

# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)