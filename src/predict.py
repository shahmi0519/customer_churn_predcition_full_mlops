# src/predict.py
import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

def load_model(path: str = MODEL_PATH):
    """Load the saved pipeline (preprocessing + model)."""
    return joblib.load(path)

def predict(df: pd.DataFrame):
    """
    Predict churn for a dataframe.
    Expects columns:
    - Product Price, Quantity, Total Purchase Amount, Customer Age, Returns
    - Product Category, Payment Method, Gender
    """
    model = load_model()
    predictions = model.predict(df)
    return predictions

if __name__ == "__main__":
    # Example usage with CSV
    df = pd.read_csv("data/raw/ecommerce_customer.csv")
    # Drop non-feature columns
    df = df.drop(columns=["Customer ID", "Customer Name", "Purchase Date", "Age"], errors="ignore")
    
    preds = predict(df)
    df["Churn_Prediction"] = preds
    print(df[["Churn_Prediction"]].head())