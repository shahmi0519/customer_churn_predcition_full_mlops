# tests/test_pipeline.py
import os
from src.pipeline import run_pipeline

def test_training_pipeline():
    DATA_PATH = "data/raw/ecommerce_customer.csv"
    TARGET = "Churn"

    NUM_COLS = [
        "Product Price",
        "Quantity",
        "Total Purchase Amount",
        "Customer Age",
        "Returns"
    ]

    CAT_COLS = [
        "Product Category",
        "Payment Method",
        "Gender"
    ]

    metrics = run_pipeline(
        data_path=DATA_PATH,
        target=TARGET,
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS
    )

    # Check metrics keys
    assert "accuracy" in metrics
    assert "report" in metrics
    # Check model file
    assert os.path.exists("models/model.pkl")