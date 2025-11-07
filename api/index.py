from flask import Flask, request, jsonify, redirect, url_for
from flasgger import Swagger
import pandas as pd
import numpy as np
import joblib
import os
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore, auth

# ============ Logging Setup ============
# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# File handler for all logs
file_handler = RotatingFileHandler(
    'logs/salefney_api.log',
    maxBytes=10000000,  # 10MB
    backupCount=5
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# File handler for errors only
error_file_handler = RotatingFileHandler(
    'logs/error.log',
    maxBytes=10000000,  # 10MB
    backupCount=5
)
error_file_handler.setFormatter(formatter)
error_file_handler.setLevel(logging.ERROR)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Create logger
logger = logging.getLogger('salefney_api')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(error_file_handler)
logger.addHandler(console_handler)

# ============ Flask Setup ============
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Firebase Configuration
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate("salefney-shokran-firebase-adminsdk-fbsvc-2413740965.json")  # Make sure to place your service account key file
    firebase_admin.initialize_app(cred)

db = firestore.client()
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key

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
    logger.info('Received prediction request')
    """
    Make a prediction and save it to database
    ---
    tags:
      - Prediction
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token for user authentication (optional)
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
        logger.info('Building feature row for prediction')
        X_row, raw = build_feature_row(data)
    except Exception as e:
        logger.error(f'Error building feature row: {str(e)}', exc_info=True)
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

    # Prepare prediction result
    current_time = datetime.utcnow()
    prediction_result = {
        "model": "xgb_model",
        "score": round(score, 4),
        "prediction": label,
        "inputs": raw,
        "categorical": {k: data.get(k) for k in GROUPS.keys()},
        "timestamp": current_time,
        "created_at": current_time.isoformat() + 'Z',  # ISO format string for better compatibility
        "user_id": None  # Will be populated if user is authenticated
    }

    # Get user ID from JWT token if available
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        try:
            token = auth_header.split(' ')[1]
            # Verify JWT token
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            prediction_result['user_id'] = payload['uid']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired. Please login again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token. Please login again."}), 401
        except Exception as e:
            # Other verification errors, continue without user_id
            print(f"Token verification failed: {str(e)}")
            pass

    try:
        # Save prediction to Firestore
        logger.info('Saving prediction to Firestore')
        db.collection('predictions').add(prediction_result)
        logger.info('Prediction saved successfully')

        # Return prediction response
        return jsonify(prediction_result)
    except Exception as e:
        logger.error(f'Error saving prediction: {str(e)}', exc_info=True)
        return jsonify({"error": "Failed to save prediction", "detail": str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    logger.info('Received request to get predictions')
    """
    Get prediction history
    ---
    tags:
      - Prediction History
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token required when user_only=true
      - name: user_only
        in: query
        type: boolean
        required: false
        default: false
        description: If true, returns only predictions for the authenticated user
      - name: limit
        in: query
        type: integer
        required: false
        default: 10
        description: Maximum number of predictions to return
    responses:
      200:
        description: List of predictions
        schema:
          type: array
          items:
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
              categorical:
                type: object
                properties:
                  Grade:
                    type: string
                    example: "B"
                  Residence:
                    type: string
                    example: "RENT"
              timestamp:
                type: string
                format: date-time
                example: "2025-11-07T12:00:00Z"
              user_id:
                type: string
                example: "507f1f77bcf86cd799439011"
      401:
        description: Unauthorized (when requesting user-specific predictions without authentication)
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Authentication required"
    """
    limit = int(request.args.get('limit', 10))
    user_only = request.args.get('user_only', 'false').lower() == 'true'

    if user_only:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        logger.info('Checking authentication for user-specific predictions')
        
        if not auth_header:
            logger.warning('Authentication failed: No Authorization header present')
            return jsonify({"error": "Authentication required - No Authorization header"}), 401
            
        if not auth_header.startswith('Bearer '):
            logger.warning('Authentication failed: Invalid Authorization header format')
            return jsonify({"error": "Authentication required - Invalid header format. Use 'Bearer <token>'"}), 401

        try:
            token = auth_header.split(' ')[1]
            logger.info('Verifying JWT token')
            # Verify JWT token
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload['uid']
            logger.info(f'Successfully authenticated user: {payload.get("email")}')
            
            # Query Firestore for user's predictions
            try:
                predictions_ref = db.collection('predictions')
                # First, get documents filtered by user_id
                query = predictions_ref.where('user_id', '==', user_id)
                docs = query.get()
                
                # Convert to list and sort in memory
                predictions = [doc.to_dict() for doc in docs]
                predictions.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                
                # Apply limit after sorting
                predictions = predictions[:limit]
                
                logger.info(f'Successfully retrieved {len(predictions)} predictions for user')
                
            except Exception as e:
                logger.error(f'Firestore query error: {str(e)}', exc_info=True)
                return jsonify({
                    "error": "Failed to retrieve predictions",
                    "detail": str(e),
                    "code": "QUERY_ERROR"
                }), 500
        except jwt.ExpiredSignatureError:
            logger.warning('Authentication failed: Token has expired')
            return jsonify({
                "error": "Token has expired. Please login again.",
                "code": "TOKEN_EXPIRED"
            }), 401
        except jwt.InvalidTokenError:
            logger.warning('Authentication failed: Invalid token format or signature')
            return jsonify({
                "error": "Invalid token. Please login again.",
                "code": "INVALID_TOKEN"
            }), 401
        except Exception as e:
            logger.error(f'Authentication failed with unexpected error: {str(e)}', exc_info=True)
            return jsonify({
                "error": "Authentication failed: " + str(e),
                "code": "AUTH_ERROR"
            }), 401
    else:
        # Get all predictions
        try:
            predictions_ref = db.collection('predictions')
            # Get all documents
            docs = predictions_ref.get()
            
            # Convert to list and sort in memory
            predictions = [doc.to_dict() for doc in docs]
            predictions.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            # Apply limit after sorting
            predictions = predictions[:limit]
            
            logger.info(f'Successfully retrieved {len(predictions)} predictions')
            
        except Exception as e:
            logger.error(f'Firestore query error: {str(e)}', exc_info=True)
            return jsonify({
                "error": "Failed to retrieve predictions",
                "detail": str(e),
                "code": "QUERY_ERROR"
            }), 500

    return jsonify(predictions)

# ============ User Authentication Endpoints ============
@app.route('/register', methods=['POST'])
def register():
    logger.info('Received registration request')
    """
    Register a new user
    ---
    tags:
      - Authentication
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - email
            - username
            - password
          properties:
            email:
              type: string
              example: "user@example.com"
              description: User's email address (must be unique)
            username:
              type: string
              example: "username"
              description: User's username (must be unique)
            password:
              type: string
              example: "password123"
              description: User's password (will be securely hashed)
    responses:
      201:
        description: User successfully registered
        schema:
          type: object
          properties:
            message:
              type: string
              example: "User registered successfully"
      400:
        description: Bad request (missing fields or duplicate email/username)
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Email already exists"
    """
    data = request.get_json()
    logger.info('Processing registration data')

    if not data or not data.get('email') or not data.get('password') or not data.get('username'):
        logger.warning('Registration failed: Missing required fields')
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        # Check if username exists in Firestore
        users_ref = db.collection('users')
        username_query = users_ref.where('username', '==', data['username']).limit(1).get()
        if len(list(username_query)) > 0:
            logger.warning(f'Registration failed: Username {data["username"]} already exists')
            return jsonify({'error': 'Username already exists'}), 400

        # Create user in Firebase Auth with email verification enabled
        user_record = auth.create_user(
            email=data['email'],
            password=data['password'],
            email_verified=False,  # User will need to verify their email
            display_name=data['username']  # Set display name to username
        )

        # Store additional user data in Firestore
        user_data = {
            'email': data['email'],
            'username': data['username'],
            'created_at': datetime.utcnow(),
            'uid': user_record.uid
        }

        # Save user data to Firestore
        db.collection('users').document(user_record.uid).set(user_data)
        
        return jsonify({'message': 'User registered successfully'}), 201
    except auth.EmailAlreadyExistsError:
        return jsonify({'error': 'Email already exists'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/login', methods=['POST'])
def login():
    logger.info('Received login request')
    """
    User login
    ---
    tags:
      - Authentication
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - email
            - password
          properties:
            email:
              type: string
              example: "user@example.com"
              description: User's registered email address
            password:
              type: string
              example: "password123"
              description: User's password
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            token:
              type: string
              description: JWT token to be used for authenticated requests
              example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            username:
              type: string
              description: User's username
              example: "username"
            email:
              type: string
              description: User's email
              example: "user@example.com"
      400:
        description: Bad request (missing email or password)
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Missing email or password"
      401:
        description: Invalid credentials
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Invalid email or password"
    """
    data = request.get_json()
    logger.info('Processing login data')

    if not data or not data.get('email') or not data.get('password'):
        logger.warning('Login failed: Missing email or password')
        return jsonify({'error': 'Missing email or password'}), 400

    try:
        # Get user by email
        try:
            user = auth.get_user_by_email(data['email'])
            logger.info(f'User found: {data["email"]}')
        except auth.UserNotFoundError:
            logger.warning(f'Login failed: Invalid credentials for email {data["email"]}')
            return jsonify({'error': 'Invalid email or password'}), 401

        # Get additional user data from Firestore
        user_doc = db.collection('users').document(user.uid).get()
        if not user_doc.exists:
            return jsonify({'error': 'User data not found'}), 404

        user_data = user_doc.to_dict()

        # Create a JWT token with user information
        token = jwt.encode({
            'uid': user.uid,
            'email': user.email,
            'username': user_data.get('username'),
            'exp': datetime.utcnow() + timedelta(days=1)  # Token expires in 1 day
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            'token': token,
            'username': user_data.get('username'),
            'email': user.email,
            'uid': user.uid
        }), 200

    except auth.UserNotFoundError:
        return jsonify({'error': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 401

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
# ============ Error Handlers ============
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f'Unhandled error: {str(error)}', exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500

# ============ Run Flask App ============
if __name__ == '__main__':
    logger.info('Starting Salefney API server...')
    app.run(debug=True)