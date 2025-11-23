import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model(processed_dir="data/processed/"):
    X_train = pd.read_csv(processed_dir + "X_train.csv")
    X_test = pd.read_csv(processed_dir + "X_test.csv")
    y_train = pd.read_csv(processed_dir + "y_train.csv")["Ad_Performance"]
    y_test = pd.read_csv(processed_dir + "y_test.csv")["Ad_Performance"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nMODEL REPORT:\n")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ad_model.pkl")
    print("\nModel saved to models/ad_model.pkl")

if __name__ == "__main__":
    train_model()
