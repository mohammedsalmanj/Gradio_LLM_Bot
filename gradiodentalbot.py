import subprocess, sys

# Auto-install required packages
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

packages = [
    "langchain",
    "langchain-community",
    "transformers",
    "torch",
    "pypdf",
    "faiss-cpu",
    "gradio"
]

for pkg in packages:
    try:
        __import__(pkg.split("==")[0].replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

# Imports
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
import gradio as gr

# Load a local LLM (free)
def get_llm():
    return pipeline("text-generation", model="distilgpt2", device=-1)

# Load PDF
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Split text into chunks
def text_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Create FAISS vector database
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embedding_model)

# Full RAG pipeline
def retriever_qa(file, query):
    llm_pipeline = get_llm()
    docs = document_loader(file.name)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()

    # Simulate retrieval + generation (local, no API)
    relevant_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in relevant_docs])
    prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {query}\nAnswer:"
    output = llm_pipeline(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    return output

# Gradio UI
rag_app = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Your Question")
    ],
    outputs="text",
    title="Local PDF Q&A Bot",
    description="Upload a PDF and ask questions locally without any account."
)

if __name__ == "__main__":
    rag_app.launch()
