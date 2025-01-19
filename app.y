import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_pipeline import ModelPipeline
from src.inference import Inference

# Load preprocessed data
@st.cache_data
def load_data():
    #raw_data = pd.read_csv("data/raw/acris_real_property_master.csv")
    processed_data = pd.read_csv("data/processed/acris_real_property_master_processed.csv")
    return processed_data

# Initialize models
@st.cache_resource
def load_models():
    model_pipeline = ModelPipeline()
    inference = Inference(model_name="EleutherAI/gpt-neo-125M")  # Smaller model
    return model_pipeline, inference

# Function to display EDA visualizations
def show_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Distribution of BOROUGH
    st.write("### Distribution of Properties by Borough")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="BOROUGH")
    st.pyplot(plt)

    # Distribution of DOC. TYPE
    st.write("### Distribution of Document Types")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="DOC. TYPE")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Top 10 most common document types
    st.write("### Top 10 Document Types")
    top_doc_types = df["DOC. TYPE"].value_counts().nlargest(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_doc_types.index, y=top_doc_types.values)
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Main function
def main():
    st.title("NYC Real Estate AI")
    st.write("Ask questions about NYC real estate properties and explore the data.")

    # Load data and models
    processed_data = load_data()
    model_pipeline, inference = load_models()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Home", "Raw Data", "Processed Data", "EDA", "Query Property"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.write("Welcome to the NYC Real Estate AI app! Use the sidebar to navigate.")

    # elif choice == "Raw Data":
    #     st.subheader("Raw Data")
    #     st.write("This is the raw ACRIS dataset.")
    #     st.dataframe(raw_data)

    elif choice == "Processed Data":
        st.subheader("Processed Data")
        st.write("This is the cleaned and preprocessed ACRIS dataset.")
        st.dataframe(processed_data)

    elif choice == "EDA":
        st.subheader("Exploratory Data Analysis (EDA)")
        show_eda(processed_data)

    elif choice == "Query Property":
        st.subheader("Query Property Information")
        query = st.text_input("Enter your query (e.g., 'Who owns the property at 123 Main Street?'):")

        if query:
            # Generate embeddings and build FAISS index
            embeddings = model_pipeline.generate_embeddings(processed_data["DOC. TYPE"].tolist())
            index = model_pipeline.build_faiss_index(embeddings)

            # Retrieve documents and generate response
            retrieved_docs = model_pipeline.retrieve_documents(query, index, processed_data)
            context = "\n".join(retrieved_docs["DOC. TYPE"].tolist())
            response = inference.generate_response(query, context)

            # Display top retrieval results
            st.subheader("Top Retrieved Documents")
            st.dataframe(retrieved_docs[["BOROUGH", "DOC. TYPE", "RECORDED / FILED"]])

            # Display inference output
            st.subheader("AI Response")
            st.write(response)

if __name__ == "__main__":
    main()