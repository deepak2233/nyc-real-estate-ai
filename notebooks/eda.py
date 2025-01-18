import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a 'plots' directory if it doesn't exist
plots_dir = "notebooks/plots"
os.makedirs(plots_dir, exist_ok=True)

# Load processed data
df = pd.read_csv("data/processed/acris_real_property_master_processed.csv")

# Display basic info
print("Data Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 Rows:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Distribution of BOROUGH
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="BOROUGH")
plt.title("Distribution of Properties by Borough")
plt.xlabel("Borough")
plt.ylabel("Count")
plt.savefig(f"{plots_dir}/borough_distribution.png")  # Save the plot
plt.show()

# Distribution of DOC. TYPE
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="DOC. TYPE")
plt.title("Distribution of Document Types")
plt.xlabel("Document Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{plots_dir}/doc_type_distribution.png")  # Save the plot
plt.show()

# Top 10 most common document types
top_doc_types = df["DOC. TYPE"].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_doc_types.index, y=top_doc_types.values)
plt.title("Top 10 Document Types")
plt.xlabel("Document Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{plots_dir}/top_10_doc_types.png")  # Save the plot
plt.show()