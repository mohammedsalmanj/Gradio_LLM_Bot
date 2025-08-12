# pip install chromadb sentence-transformers

import chromadb
from chromadb.utils import embedding_functions

# Define embedding model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Chroma client
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="employee_collection",
    metadata={"description": "A collection for storing employee data"},
    configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef}
)

# Employee data
employees = [
    {
        "id": "employee_1",
        "name": "John Doe",
        "experience": 5,
        "department": "Engineering",
        "role": "Software Engineer",
        "skills": ["Python", "JavaScript", "React", "Node.js", "databases"],
        "location": "New York",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_2",
        "name": "Jane Smith",
        "experience": 8,
        "department": "Marketing",
        "role": "Marketing Manager",
        "skills": ["Digital marketing", "SEO", "content strategy", "analytics", "social media"],
        "location": "Los Angeles",
        "employment_type": "Full-time"
    },
]

# Convert employee info into documents
employee_documents = [
    f"{emp['role']} with {emp['experience']} years of experience in {emp['department']}. "
    f"Skills: {', '.join(emp['skills'])}. Located in {emp['location']}. "
    f"Employment type: {emp['employment_type']}."
    for emp in employees
]

# Add data to ChromaDB
collection.add(
    ids=[emp["id"] for emp in employees],
    documents=employee_documents,
    metadatas=[{
        "name": emp["name"],
        "department": emp["department"],
        "role": emp["role"],
        "experience": emp["experience"],
        "location": emp["location"],
        "employment_type": emp["employment_type"]
    } for emp in employees]
)

# Search example
query_text = "Python developer with web development experience"
results = collection.query(query_texts=[query_text], n_results=2)

print(f"Query: {query_text}")
for i, (doc_id, doc, dist) in enumerate(zip(
    results["ids"][0], results["documents"][0], results["distances"][0]
)):
    meta = results["metadatas"][0][i]
    print(f"{i+1}. {meta['name']} ({doc_id}) - Distance: {dist:.4f}")
    print(f"   Role: {meta['role']} | Location: {meta['location']}")
    print(f"   Document: {doc}")
