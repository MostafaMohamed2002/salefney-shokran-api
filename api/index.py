from flask import Flask, request, jsonify, redirect, url_for
from flasgger import Swagger
import pandas as pd
import numpy as np
import joblib
import os
from flask_cors import CORS
# ============ Flask Setup ============
app = Flask(__name__)
CORS(app) 
swagger = Swagger(app)

# ============ Load Model ============
MODEL_PATH = "models/xgb_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

saved = joblib.load("models/xgb_model.pkl")
model = saved["model"]
feature_order = saved["feature_names"]
print(f"✅ Loaded model from {MODEL_PATH}")

# Optional scaler
scaler = None

# ============ Feature Definitions ============
# Categorical one-hot groups (exactly like GUI)
GROUPS = {
    "Grade": ["A", "B", "C", "D", "E", "F", "G"],
    "Residence": ["MORTGAGE", "RENT", "OWN", "OTHER"],
    "Purpose": [
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
        "MEDICAL", "PERSONAL", "VENTURE"
    ],
    "Age_band": ["20-25", "26-35", "36-45", "46-55", "56-65"],
    "Size_band": ["small", "medium", "large", "very large"],
    "Binary_flag": ["Y", "N"],
    "Band": ["high", "high-middle", "middle", "low-middle", "low"]
}

# Numeric raw inputs
NUMERIC_FIELDS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate"
]

# ============ Helper: Build Feature Row ============
def build_feature_row(data: dict):
    """
    Builds full feature vector from GUI-like JSON input
    """
    # ----- 1️⃣ start from categorical -----
    all_features = {}
    for group_name, options in GROUPS.items():
        selected = data.get(group_name)
        for opt in options:
            all_features[opt] = 1 if opt == selected else 0

    # ----- 2️⃣ numeric raw -----
    vals = {
        "person_age": float(data.get("person_age", 30.0)),
        "person_income": float(data.get("person_income", 30000.0)),
        "person_emp_length": float(data.get("person_emp_length", 3.0)),
        "loan_amnt": float(data.get("loan_amnt", 10000.0)),
        "loan_int_rate": float(data.get("loan_int_rate", 12.0)) / 100.0,
    }

    # Derived metrics
    denom_income = max(1.0, vals["person_income"])
    loan_pct = vals["loan_amnt"] / denom_income
    cred_hist_len = 5.0
    loan_to_emp = vals["loan_amnt"] / max(1.0, vals["person_emp_length"])
    int_to_loan = vals["loan_int_rate"] / max(1.0, vals["loan_amnt"])

    raw_fill = {
        **vals,
        "loan_percent_income": loan_pct,
        "cb_person_cred_hist_length": cred_hist_len,
        "loan_to_income_ratio": loan_pct,
        "loan_to_emp_length_ratio": loan_to_emp,
        "int_rate_to_loan_amt_ratio": int_to_loan,
    }

    all_features.update(raw_fill)
    X = pd.DataFrame([all_features])

    # Optional scaling
    if scaler is not None:
        common = [c for c in getattr(scaler, "feature_names_in_", []) if c in X.columns]
        for c in common:
            j = list(scaler.feature_names_in_).index(c)
            m = float(scaler.mean_[j])
            s = float(scaler.scale_[j]) or 1.0
            X[c] = (X[c] - m) / s

    return X, raw_fill

# ============ Prediction Function ============
def predict_score(m, X):
    if hasattr(m, "predict_proba"):
        return float(m.predict_proba(X)[:, -1][0])
    if hasattr(m, "decision_function"):
        z = float(m.decision_function(X)[0])
        return 1 / (1 + np.exp(-z))
    return float(m.predict(X)[0])

# ============ API Endpoints ============
@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction (identical to Mini GUI)
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            person_age:
              type: number
              example: 30
            person_income:
              type: number
              example: 30000
            person_emp_length:
              type: number
              example: 3
            loan_amnt:
              type: number
              example: 10000
            loan_int_rate:
              type: number
              example: 12
            Grade:
              type: string
              enum: ["A","B","C","D","E","F","G"]
              example: "B"
            Residence:
              type: string
              enum: ["MORTGAGE","RENT","OWN","OTHER"]
              example: "RENT"
            Purpose:
              type: string
              enum: ["DEBTCONSOLIDATION","EDUCATION","HOMEIMPROVEMENT","MEDICAL","PERSONAL","VENTURE"]
              example: "PERSONAL"
            Age_band:
              type: string
              enum: ["20-25","26-35","36-45","46-55","56-65"]
              example: "26-35"
            Size_band:
              type: string
              enum: ["small","medium","large","very large"]
              example: "medium"
            Binary_flag:
              type: string
              enum: ["Y","N"]
              example: "Y"
            Band:
              type: string
              enum: ["high","high-middle","middle","low-middle","low"]
              example: "middle"
    responses:
      200:
        description: Successful prediction result (same as Mini GUI)
        schema:
          type: object
          properties:
            model:
              type: string
              example: "xgb_model"
            score:
              type: number
              format: float
              example: 0.3421
            prediction:
              type: string
              example: "its safe"
            inputs:
              type: object
              properties:
                person_age:
                  type: number
                  example: 35.0
                person_income:
                  type: number
                  example: 50000.0
                person_emp_length:
                  type: number
                  example: 5.0
                loan_amnt:
                  type: number
                  example: 12000.0
                loan_int_rate:
                  type: number
                  example: 0.105
                loan_percent_income:
                  type: number
                  example: 0.24
                cb_person_cred_hist_length:
                  type: number
                  example: 5.0
                loan_to_income_ratio:
                  type: number
                  example: 0.24
                loan_to_emp_length_ratio:
                  type: number
                  example: 2400.0
                int_rate_to_loan_amt_ratio:
                  type: number
                  example: 8.75e-06
            categorical:
              type: object
              properties:
                Grade:
                  type: string
                  example: "B"
                Residence:
                  type: string
                  example: "RENT"
                Purpose:
                  type: string
                  example: "PERSONAL"
                Age_band:
                  type: string
                  example: "26-35"
                Size_band:
                  type: string
                  example: "medium"
                Binary_flag:
                  type: string
                  example: "Y"
                Band:
                  type: string
                  example: "middle"
    """
    # Accept JSON body or form data
    data = request.get_json(silent=True)
    if data is None:
        # fallback to form-encoded data
        data = request.form.to_dict() if request.form else None

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Build feature vector and the raw inputs dictionary
    try:
        X_row, raw = build_feature_row(data)
    except Exception as e:
        return jsonify({"error": "invalid input", "detail": str(e)}), 400

    # Determine expected feature order (prefer saved feature_order if present)
    expected_features = None
    if 'feature_order' in globals() and feature_order:
        expected_features = feature_order
    else:
        try:
            expected_features = model.get_booster().feature_names
        except Exception:
            expected_features = X_row.columns.tolist()

    # Ensure X_row has exactly the expected columns (missing filled with 0)
    X_row = X_row.reindex(columns=expected_features, fill_value=0)

    score = predict_score(model, X_row)
    pred = int(score >= 0.5)
    label = {0: "its safe", 1: "High Risk"}[pred]

    return jsonify({
        "model": "xgb_model",
        "score": round(score, 4),
        "prediction": label,
        "inputs": raw,
        "categorical": {k: data.get(k) for k in GROUPS.keys()}
    })

# ============ Redirect Root to Swagger ============
@app.route('/')
def home():
    return redirect(url_for('flasgger.apidocs'))