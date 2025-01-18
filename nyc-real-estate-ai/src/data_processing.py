import dask.dataframe as dd
from tqdm import tqdm

def load_csv_with_dask(csv_path: str) -> dd.DataFrame:
    """
    Load data using Dask.
    """
    print(f"Loading data from {csv_path} using Dask...")
    df = dd.read_csv(csv_path, usecols=["BOROUGH", "DOC. TYPE", "RECORDED / FILED"])
    print("Data loaded successfully.")
    return df

def preprocess_data(df: dd.DataFrame, text_column: str = "DOC. TYPE") -> dd.DataFrame:
    """
    Preprocess the data.
    """
    print("Preprocessing data...")
    # Fill missing values
    df[text_column] = df[text_column].fillna("")
    # Convert text to lowercase
    df[text_column] = df[text_column].str.lower()
    print("Data preprocessing complete.")
    return df

def save_processed_data(df: dd.DataFrame, output_path: str) -> None:
    """
    Save the preprocessed data to a CSV file.
    """
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False, single_file=True)  # Save as a single file
    print(f"Data successfully saved to {output_path}.")

# Example usage
if __name__ == "__main__":
    # Load and preprocess real property master data
    csv_path = "data/raw/acris_real_property_master.csv"
    df = load_csv_with_dask(csv_path)
    df = preprocess_data(df)

    # Save processed data
    output_path = "data/processed/acris_real_property_master_processed.csv"
    save_processed_data(df, output_path)