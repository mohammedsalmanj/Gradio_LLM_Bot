# pip install chromadb sentence-transformers

import chromadb
from chromadb.utils import embedding_functions

def main():
    # Setup embedding function (local, open-source)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create local Chroma client (no cloud needed)
    client = chromadb.Client()

    # Create or get collection
    collection = client.create_collection(
        name="local_grocery_collection",
        metadata={"description": "Local grocery data"},
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": ef
        }
    )

    # Sample grocery texts
    texts = [
        'fresh red apples',
        'organic bananas',
        'ripe mangoes',
        'whole wheat bread',
        'farm-fresh eggs',
    ]

    # Unique IDs for documents
    ids = [f"item_{i+1}" for i in range(len(texts))]

    # Add data to collection (embeddings generated automatically)
    collection.add(
        documents=texts,
        metadatas=[{"category": "food"} for _ in texts],
        ids=ids
    )

    # Query example
    query_term = "apple"
    results = collection.query(query_texts=[query_term], n_results=3)

    # Print results
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