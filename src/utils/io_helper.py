import pandas as pd

def load_csv(path: str):
    """Load a CSV file safely."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None

def save_csv(df, path: str):
    """Save a CSV file."""
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def row_to_dict(df, row_index: int):
    """Convert a DataFrame row into a dict for Gemini prompts."""
    return df.iloc[row_index].to_dict()
