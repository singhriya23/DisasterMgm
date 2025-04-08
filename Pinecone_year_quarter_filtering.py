import os
import json
import re
from pinecone import Pinecone, ServerlessSpec  
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# ✅ Load .env once at the top
load_dotenv(dotenv_path=".env")


def extract_quarter_year(text):
    """Extracts Quarter and Year from a chunk of text using regex."""
    match = re.search(r"(Q[1-4])\s*[,:\-]?\s*(\d{4})", text, re.IGNORECASE)
    if match:
        return match.group(2), match.group(1).upper()  # Returns (Year, Quarter)
    return None, None


def index_json_content(json_content, index_name="json-index", region="us-east-1"):
    """
    Index JSON content into Pinecone with Year and Quarter as structured metadata.

    Args:
        json_content (str or dict): The JSON content containing chunks.
        index_name (str): Pinecone index name.
        region (str): Pinecone region.

    Returns:
        PineconeVectorStore: The indexed vector store.
    """
    # Ensure index name is Pinecone-compliant
    index_name = index_name.lower().replace("_", "-")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("❌ Pinecone API Key is missing!")

    # ✅ Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # ✅ Check if index exists
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )

    # ✅ Load the Pinecone index
    index = pc.Index(index_name)

    # ✅ Load Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Initialize Vector Store
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="page_content")

    # ✅ Parse JSON
    try:
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format.")

    # ✅ Validate 'chunks' list
    if "chunks" not in data or not isinstance(data["chunks"], list):
        raise ValueError("❌ No valid chunks found in the JSON content.")

    chunks = [chunk if isinstance(chunk, str) else chunk.get("content", "") for chunk in data["chunks"]]

    # ✅ Convert chunks to LangChain Documents with metadata
    documents = []
    for chunk in chunks:
        if chunk.strip():
            year, quarter = extract_quarter_year(chunk)
            metadata = {
                
                "year": year or "unknown",
                "quarter": quarter or "unknown"
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    if not documents:
        print("⚠️ No valid chunks found after processing.")
        return None

    # ✅ Index documents into Pinecone
    vector_store.add_documents(documents)
    print(f"✅ Successfully indexed {len(documents)} chunks with Year & Quarter metadata into Pinecone ({index_name}).")

    return vector_store


# Example Usage
if __name__ == "__main__":
    json_file_path = "output-json/output-q1-2021.json"  # Ensure this file exists
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_content = f.read()

    vector_store = index_json_content(json_content, index_name="json-index-1", region="us-east-1")
    if vector_store:
        print("✅ Indexing completed successfully.")
    else:
        print("❌ Indexing failed.")