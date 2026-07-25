
# ============================================================
# SuperKart Sales Prediction API
# Flask application for serving ML model predictions
# ============================================================


# Import libraries for data handling
import pandas as pd
import joblib

# Import Flask components for creating API endpoints
from flask import Flask, request, jsonify

# Import logging utilities for monitoring API execution
import logging
import sys
import os


# ============================================================
# Initialize Flask Application
# ============================================================

# Create Flask application instance
superkart_api = Flask("superkart_sales_app")


# ============================================================
# Configure Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

logger.info(f"Module name: {__name__}")
logger.info(f"Flask application name: {superkart_api.name}")
logger.info(f"Application root path: {superkart_api.root_path}")


# ============================================================
# Load Trained Machine Learning Model
# ============================================================

# Define model location
model_path = "superkart_sales_model.joblib"

# Load serialized ML pipeline
# The pipeline includes preprocessing + trained regression model
model = joblib.load(model_path)

logger.info("SuperKart sales prediction model loaded successfully")


# ============================================================
# Home Endpoint
# ============================================================

@superkart_api.route("/", methods=["GET"])
def home():
    """
    Root endpoint to verify that the API is running.

    Returns:
        HTML welcome message with API usage instructions.
    """

    logger.info("Home endpoint accessed")

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SuperKart Sales Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
                background-color: #f4f4f4;
            }

            h1 {
                color: #333;
                font-size: 3em;
            }

            p {
                color: #666;
                font-size: 1.5em;
            }
        </style>
    </head>

    <body>
        <h1>Welcome to SuperKart Sales Prediction API</h1>
        <p>
        Send a POST request to 
        <b>/v1/predict</b>
        to get sales predictions.
        </p>
    </body>

    </html>
    """

    return html



# ============================================================
# Prediction Endpoint
# ============================================================

@superkart_api.route("/v1/predict", methods=["POST"])
def predict_sales():
    """
    Accepts product and store details in JSON format
    and returns predicted sales value.

    Request:
        JSON input containing model features.

    Response:
        Predicted sales value in JSON format.
    """

    try:

        # Read JSON payload from API request
        data = request.get_json()

        logger.info("Prediction request received")


        # Extract required features from input JSON
        sample = {

            "Product_Weight": data["Product_Weight"],

            "Product_Sugar_Content": data["Product_Sugar_Content"],

            "Product_Allocated_Area": data["Product_Allocated_Area"],

            "Product_MRP": data["Product_MRP"],

            "Store_Size": data["Store_Size"],

            "Store_Location_City_Type": data["Store_Location_City_Type"],

            "Store_Type": data["Store_Type"],

            "Store_Age_Years": data["Store_Age_Years"],

            "Product_Type_Category": data["Product_Type_Category"],

            "Product_Id_char": data["Product_Id_char"]

        }


        # Convert input dictionary into DataFrame
        # Required because the ML pipeline expects tabular input
        input_data = pd.DataFrame([sample])


        logger.info(f"Input data received:\n{input_data}")


        # Generate prediction using trained pipeline
        prediction = model.predict(input_data)[0]


        logger.info(f"Prediction generated: {prediction}")


        # Return prediction as JSON response
        return jsonify(
            {
                "Predicted_Product_Store_Sales_Total": prediction
            }
        )


    except KeyError as e:

        # Handle missing input features
        logger.error(f"Missing input key: {e}")

        return jsonify(
            {
                "error": f"Missing key: {str(e)}"
            }
        ), 400


    except Exception as e:

        # Handle unexpected errors
        logger.error(f"Prediction failed: {e}")

        return jsonify(
            {
                "error": f"Prediction failed: {str(e)}"
            }
        ), 500



# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    # Start Flask development server
    superkart_api.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
