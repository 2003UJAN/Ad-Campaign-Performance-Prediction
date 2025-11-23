import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data(input_path, output_dir="data/processed/"):
    df = pd.read_csv(input_path)

    categorical_cols = ["Gender"]
    label_encoder = LabelEncoder()
    df["Gender"] = label_encoder.fit_transform(df["Gender"])

    y = df["Ad_Performance"]
    X = df.drop("Ad_Performance", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=X.columns).to_csv(output_dir + "X_train.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(output_dir + "X_test.csv", index=False)
    y_train.to_csv(output_dir + "y_train.csv", index=False)
    y_test.to_csv(output_dir + "y_test.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print("Data preprocessing completed.")
