## 1. Project Overview
The goal of this project is to create an AI-powered tool for real estate professionals that can:

- Retrieve relevant property information from the ACRIS dataset.
- Generate human-readable answers to user queries (e.g., "Who owns this property?" or "What permits are required for renovations?").

The pipeline consists of the following steps:

1. **Data Download:** Download the ACRIS dataset and load it into an SQLite database.
2. **Data Preprocessing:** Clean and preprocess the data for analysis.
3. **Exploratory Data Analysis (EDA):** Explore the dataset to understand its structure and content.
4. **RAG Pipeline:**
   - Generate embeddings for property descriptions.
   - Build a FAISS index for efficient retrieval.
   - Integrate with a language model (e.g., Mistral-7B) to generate responses.
5. **Inference:** Query the pipeline and generate actionable insights.

---

## 2. Setup Instructions
### Prerequisites
- Python 3.8 or higher.
- A system with sufficient RAM (at least 8 GB recommended).

### Install Dependencies
Clone the repository:
```bash
git clone https://github.com/yourusername/nyc-real-estate-ai.git
cd nyc-real-estate-ai
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## 3. Data Download and Processing
### Download the ACRIS Dataset
The ACRIS dataset is available as CSV files. You can download it using the `nycdb` tool:

Install `nycdb`:
```bash
pip install nycdb
```

Download and load the ACRIS dataset into an SQLite database:
```bash
nycdb --database sqlite:///data/nycdb.sqlite --root-dir data/raw --download acris --load acris
```
This will:
- Download the ACRIS dataset.
- Load the data into the SQLite database (`data/nycdb.sqlite`).

### Process the Data
The dataset is processed using Dask, a parallel computing library that handles large datasets efficiently. Dask is used because:
- It processes data in chunks, reducing memory usage.
- It supports lazy evaluation, which avoids loading the entire dataset into memory at once.

The preprocessing steps include:
- Loading the data in chunks.
- Cleaning the data (e.g., handling missing values, standardizing formats).
- Saving the processed data to a CSV file.

To process the data, run:
```bash
python src/data_processing.py
```

---

## 4. Pipeline Overview
### Step 1: Data Preprocessing
- **Input:** Raw ACRIS dataset (CSV files).
- **Output:** Processed data (`data/processed/acris_real_property_master_processed.csv`).
- **Tools:** Pandas, Dask.

### Step 2: Exploratory Data Analysis (EDA)
- **Purpose:** Understand the dataset's structure and content.
- **Tools:** Matplotlib, Seaborn.
- **Output:** Visualizations saved in `notebooks/plots/`.

### Step 3: Embedding Generation
- **Purpose:** Convert property descriptions into numerical embeddings for retrieval.
- **Tools:** Sentence Transformers (`all-MiniLM-L6-v2`).
- **Output:** Embeddings stored in a FAISS index.

### Step 4: FAISS Indexing
- **Purpose:** Enable efficient similarity search for retrieval.
- **Tools:** FAISS.
- **Output:** FAISS index file (`data/processed/embeddings.faiss`).

### Step 5: Language Model Integration
- **Purpose:** Generate human-readable answers based on retrieved documents.
- **Tools:** Mistral-7B (open-source language model).
- **Output:** Responses to user queries.



---

## 5. Running the Pipeline
### Step 1: Preprocess the Data
Run the data preprocessing script:
```bash
python src/data_processing.py
```

### Step 2: Perform EDA
Open the Jupyter Notebook for EDA:
```bash
jupyter notebook notebooks/eda.ipynb
```

### Step 3: Run the RAG Pipeline
Run the main pipeline script with a sample query:
```bash
python main.py --query "Who owns the property at 123 Main Street?"

Loading preprocessed data from data/processed/acris_real_property_master_processed.csv...
Successfully loaded 1000 records.
Loading embedding model: all-MiniLM-L6-v2...
Embedding model loaded successfully.
Generating embeddings...
Generated embeddings of shape (1000, 384).
Building FAISS index...
FAISS index built.
Loading language model: mistralai/Mistral-7B-v0.1...
Language model loaded successfully.

=== Query ===
Who owns the property at 123 Main Street?

=== Retrieved Documents ===
   BOROUGH DOC. TYPE RECORDED / FILED
0        1      DEED        01/01/2020
1        2    MORTGAGE      02/15/2019
...

=== Response ===
The property at 123 Main Street is owned by John Doe.

```

---

## 7. Troubleshooting
### Common Issues
- **Out of Memory:**
  - Reduce the chunk size in `data_processing.py`.
  - Use a cloud environment with more RAM.
- **Missing Columns:**
  - Verify the column names in the raw dataset and update the script accordingly.
- **Model Loading Errors:**
  - Ensure the language model (e.g., Mistral-7B) is correctly installed and accessible.

### Debugging Tips
- Check the logs for error messages.
- Use smaller subsets of the dataset for testing.
- Monitor memory usage using `free -h`.

---

## 8. Future Improvements
- Fine-tune the language model on real estate-specific data.
- Add support for additional datasets (e.g., environmental data, legal records).
- Deploy the pipeline as a web application for easy access.
