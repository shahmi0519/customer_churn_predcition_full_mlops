# src/evaluate.py
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_test, y_test):
    """
    Evaluate the trained pipeline on test data.
    Returns:
    - accuracy
    - classification report dictionary
    """
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)
    return {
        "accuracy": acc,
        "report": report
    }

if __name__ == "__main__":
    # Example usage
    from src.data_loader import load_data, split_data

    DATA_PATH = "data/raw/ecommerce_customer.csv"
    TARGET = "Churn"

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, TARGET)
    
    # Example: load pipeline
    import joblib
    model = joblib.load("models/model.pkl")
    metrics = evaluate(model, X_test, y_test)
    print(metrics)