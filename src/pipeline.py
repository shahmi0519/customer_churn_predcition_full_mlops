from src.data_loader import load_data, split_data
from src.preprocessing import build_preprocessor
from src.train import train_model, save_model
from src.evaluate import evaluate
from sklearn.pipeline import Pipeline

def run_pipeline(data_path, target, num_cols, cat_cols):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target)

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train the model
    model = train_model(X_train_processed, y_train)

    # Save preprocessing + model together
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    save_model(pipeline)

    # Evaluate
    metrics = evaluate(pipeline, X_test, y_test)
    return metrics

if __name__ == "__main__":
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

    print("Training completed")
    print(metrics)