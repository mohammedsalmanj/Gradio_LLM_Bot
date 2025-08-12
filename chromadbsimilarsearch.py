# pip install chromadb sentence-transformers

import chromadb
from chromadb.utils import embedding_functions

def main():
    """
    Main function to setup and execute a local semantic search using ChromaDB and Sentence Transformers.
    It creates a collection with sample grocery data, adds documents to it, and performs a query to retrieve
    the top relevant results based on the query term.
    """

    # Setup embedding function using Sentence Transformer model
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Initialize local ChromaDB client
    client = chromadb.Client()

    # Create or retrieve a collection named 'local_grocery_collection'
    collection = client.create_collection(
        name="local_grocery_collection",
        metadata={"description": "Local grocery data"},
        configuration={
            "hnsw": {"space": "cosine"},  # Use cosine similarity for distance metric
            "embedding_function": ef
        }
    )

    # Sample grocery texts to be added to the collection
    texts = [
        'fresh red apples',
        'organic bananas',
        'ripe mangoes',
        'whole wheat bread',
        'farm-fresh eggs',
    ]

    # Generate unique IDs for each document
    ids = [f"item_{i+1}" for i in range(len(texts))]

    # Add documents to the collection; embeddings are generated automatically
    collection.add(
        documents=texts,
        metadatas=[{"category": "food"} for _ in texts],
        ids=ids
    )

    # Define query term for searching in the collection
    query_term = "apple"

    # Query the collection to find top 3 relevant documents
    results = collection.query(query_texts=[query_term], n_results=3)

    # Check and print the query results if available
    if results and results['ids'] and len(results['ids'][0]) > 0:
        print(f"Top results for query '{query_term}':")
        for i in range(len(results['ids'][0])):
            print(
                f"ID: {results['ids'][0][i]}, "
                f"Text: {results['documents'][0][i]}, "
                f"Score: {results['distances'][0][i]:.4f}"
            )
    else:
        print(f"No results found for query '{query_term}'")

if __name__ == "__main__":
    main()

# Output:
# Top results for query 'apple':
# ID: item_1, Text: fresh red apples, Score: 0.0000
# ID: item_2, Text: organic bananas, Score: 0.0000
# ID: item_3, Text: ripe mangoes, Score: 0.0000