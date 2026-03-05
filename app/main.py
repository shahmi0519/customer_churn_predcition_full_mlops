# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

# Load saved pipeline (preprocessing + model)
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

# Input schema
class CustomerData(BaseModel):
    Product_Price: float
    Quantity: int
    Total_Purchase_Amount: float
    Customer_Age: int
    Returns: float
    Product_Category: str
    Payment_Method: str
    Gender: str

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert incoming JSON to model-ready dataframe
    import pandas as pd

    df = pd.DataFrame([{
        "Product Price": data.Product_Price,
        "Quantity": data.Quantity,
        "Total Purchase Amount": data.Total_Purchase_Amount,
        "Customer Age": data.Customer_Age,
        "Returns": data.Returns,
        "Product Category": data.Product_Category,
        "Payment Method": data.Payment_Method,
        "Gender": data.Gender
    }])

    # Predict using saved pipeline
    prediction = model.predict(df)[0]

    return {
        "churn_prediction": int(prediction),
        "label": "Churn" if prediction == 1 else "No Churn"
    }