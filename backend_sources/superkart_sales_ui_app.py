from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():import os
import joblib

# Path of saved trained model
model_path = os.path.join(
    backend_sources,
    "superkart_sales_model.joblib"
)

# Check model file exists
if os.path.exists(model_path):

    # Load trained model pipeline
    saved_model = joblib.load(model_path)

    print("Model loaded successfully")
    print("Model type:", type(saved_model))

else:
    print("Model file not found:")
    print(model_path)
    return {
        "message": "SuperKart Sales Prediction API is Running"
    }
