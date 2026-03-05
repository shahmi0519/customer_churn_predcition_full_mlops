import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLUMNS = [
    "Customer ID",
    "Customer Name",
    "Purchase Date",
    "Age"  
]

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return df

def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)