import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st


# Example DataFrame
df_meta = pd.read_csv("airplane_metadata.csv", header = 0)
df_comment = pd.read_csv("airplane_comment.csv", header = 0)
df=pd.merge(df_meta, df_comment, on='ID', how='right')

df['combined_text'] = df.apply(
    lambda
        row: f"ID:  {row['ID']} WingSpan: {row['WingSpan']}  Weight: {row['weight']} FlightDistance: {row['flight_distance']} FoldDifficulty: {row['Fold difficulty']} Comment: {row['Comment']}",
    axis=1
)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the combined text
embeddings = model.encode(df['combined_text'].tolist())

np.save('airplane_embeddings.npy', embeddings)
# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, 'faiss_index.index')


def process_query(query):
    # Combine metadata and reviews into a single field for retrieval
    df['combined_text'] = df.apply(
        lambda row: f"ID:  {row['ID']} WingSpan: {row['WingSpan']}  Weight: {row['weight']} FlightDistance: {row['flight_distance']} FoldDifficulty: {row['Fold difficulty']} Comment: {row['Comment']}",
        axis=1
    )

    # Initialize Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # # Generate embeddings for the combined text
    # embeddings = model.encode(df['combined_text'].tolist())
    #
    # # Create FAISS index
    # dimension = embeddings.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    # index.add(np.array(embeddings))

    embeddings=np.load('airplane_embeddings.npy')
    index = faiss.read_index('faiss_index.index')

    # For querying
    # query = "give me the airplane that flies high."
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)  # Search for top 5 nearest neighbors

    # Retrieve the most similar products
    top_indices = I[0]
    top_products = df.iloc[top_indices]

    # Retrieve and include metadata in the final output
    top_products_with_metadata = df.iloc[top_indices]
    # print(top_products_with_metadata)
    st.write(top_products_with_metadata)

    return top_products_with_metadata

# Streamlit app layout
st.title("Please enter the query you'd like to know about finding your space craft.")

# Prompt user for input
query = st.text_input("Enter your query:")

# Button to run the function
if st.button("Run"):
    if query:  # Check if the input is not empty
        result = process_query(query)
        st.success(result)  # Display the result
    else:
        st.warning("Please enter a query before running.")
