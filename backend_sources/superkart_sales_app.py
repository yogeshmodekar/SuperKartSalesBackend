
# ============================================================
# SuperKart Sales Prediction API
# Flask Backend
# ============================================================

import os
import sys
import logging
import joblib
import pandas as pd

from flask import Flask, request, jsonify

# ============================================================
# Initialize Flask Application
# ============================================================

superkart_api = Flask(__name__)

# ============================================================
# Configure Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# ============================================================
# Load Model
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_sales_model.joblib"
)

try:

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully.")

except Exception as e:

    logger.error(f"Unable to load model : {e}")

    model = None


# ============================================================
# Home Endpoint
# ============================================================

@superkart_api.route("/", methods=["GET"])
def home():

    return jsonify({

        "Application": "SuperKart Sales Prediction API",

        "Status": "Running",

        "Version": "1.0"

    })


# ============================================================
# Health Check
# ============================================================

@superkart_api.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "Healthy",

        "model_loaded": model is not None

    })


# ============================================================
# Single Prediction
# ============================================================

@superkart_api.route("/v1/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        sample = {

            "Product_Weight":
                data["Product_Weight"],

            "Product_Sugar_Content":
                data["Product_Sugar_Content"],

            "Product_Allocated_Area":
                data["Product_Allocated_Area"],

            "Product_MRP":
                data["Product_MRP"],

            "Store_Size":
                data["Store_Size"],

            "Store_Location_City_Type":
                data["Store_Location_City_Type"],

            "Store_Type":
                data["Store_Type"],

            "Store_Age_Years":
                data["Store_Age_Years"],

            "Product_Type_Category":
                data["Product_Type_Category"],

            "Product_Id_char":
                data["Product_Id_char"]

        }

        input_df = pd.DataFrame([sample])

        prediction = float(model.predict(input_df)[0])

        return jsonify({

            "status": "success",

            "Sales": prediction

        })

    except KeyError as e:

        return jsonify({

            "status": "failed",

            "error": f"Missing field : {e}"

        }), 400

    except Exception as e:

        return jsonify({

            "status": "failed",

            "error": str(e)

        }), 500


# ============================================================
# Batch Prediction
# ============================================================

@superkart_api.route("/v1/predict_batch", methods=["POST"])
def predict_batch():

    try:

        data = request.get_json()

        records = data["records"]

        df = pd.DataFrame(records)

        predictions = model.predict(df)

        return jsonify({

            "status": "success",

            "predictions": predictions.tolist()

        })

    except Exception as e:

        return jsonify({

            "status": "failed",

            "error": str(e)

        }), 500


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    logger.info("Starting SuperKart Flask API...")

    superkart_api.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
