import numpy as np  # Add this import
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_embeddings(embeddings: np.ndarray, title: str = "Embeddings Visualization"):
    """
    Visualize embeddings using t-SNE.
    """
    print("Visualizing embeddings...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1])
    plt.title(title)
    plt.show()