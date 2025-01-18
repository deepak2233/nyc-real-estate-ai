import argparse
import pandas as pd
from src.model_pipeline import ModelPipeline
from src.inference import Inference
from src.utils import plot_embeddings

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="NYC Real Estate AI Pipeline")
    parser.add_argument("--query", type=str, required=True, help="User query (e.g., 'Who owns the property at 123 Main Street?')")
    args = parser.parse_args()

    # Step 1: Load preprocessed data
    processed_data_path = "data/processed/acris_real_property_master_processed.csv"
    print(f"Loading preprocessed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)
    print(f"Successfully loaded {len(df)} records.")

    # Step 2: Generate embeddings and build FAISS index
    model_pipeline = ModelPipeline()
    embeddings = model_pipeline.generate_embeddings(df["DOC. TYPE"].tolist())
    index = model_pipeline.build_faiss_index(embeddings)

    # Step 3: Retrieve documents and generate response
    retrieved_docs = model_pipeline.retrieve_documents(args.query, index, df)
    context = "\n".join(retrieved_docs["DOC. TYPE"].tolist())

    inference = Inference(model_name="mistralai/Mistral-7B-v0.1")  # Use an open-source model
    response = inference.generate_response(args.query, context)

    # Print results
    print("\n=== Query ===")
    print(args.query)
    print("\n=== Retrieved Documents ===")
    print(retrieved_docs[["BOROUGH", "DOC. TYPE", "RECORDED / FILED"]])
    print("\n=== Response ===")
    print(response)

    # Optional: Visualize embeddings
    plot_embeddings(embeddings)

if __name__ == "__main__":
    main()