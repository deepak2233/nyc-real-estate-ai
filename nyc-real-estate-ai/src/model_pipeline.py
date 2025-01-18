import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd  # Add this import

class ModelPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def generate_embeddings(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings of shape {embeddings.shape}.")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Build a FAISS index for efficient retrieval.
        """
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print("FAISS index built.")
        return index

    def retrieve_documents(self, query: str, index: faiss.IndexFlatL2, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Retrieve relevant documents based on a query.
        """
        query_embedding = self.model.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        return df.iloc[indices[0]]