import os
import json
import re
from pinecone import Pinecone, ServerlessSpec  
from langchain.vectorstores import PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# ✅ Load .env once at the top
load_dotenv(dotenv_path=".env")


def extract_quarter_year_from_filename(filename):
    """
    Extracts Quarter and Year from filename like 'output-q1-2021.json'
    """
    match = re.search(r"(Q[1-4])[-_]?(\d{4})", filename, re.IGNORECASE)
    if match:
        return match.group(2), match.group(1).upper()
    return None, None


def index_json_content(json_content, filename=None, index_name="json-index", region="us-east-1"):
    """
    Index JSON content into Pinecone with Year and Quarter as structured metadata from the filename.

    Args:
        json_content (str or dict): The JSON content containing chunks.
        filename (str): Optional filename to extract metadata from.
        index_name (str): Pinecone index name.
        region (str): Pinecone region.

    Returns:
        PineconeVectorStore: The indexed vector store.
    """
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

    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="page_content")

    try:
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format.")

    if "chunks" not in data or not isinstance(data["chunks"], list):
        raise ValueError("❌ No valid chunks found in the JSON content.")

    chunks = [chunk if isinstance(chunk, str) else chunk.get("content", "") for chunk in data["chunks"]]

    # ✅ Extract from filename
    year, quarter = extract_quarter_year_from_filename(filename or "")

    documents = []
    for chunk in chunks:
        if chunk.strip():
            metadata = {
                "source": filename or "in-memory",
                "year": year or "unknown",
                "quarter": quarter or "unknown"
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    if not documents:
        print("⚠️ No valid chunks found after processing.")
        return None

    vector_store.add_documents(documents)
    print(f"✅ Successfully indexed {len(documents)} chunks with Year & Quarter from filename into Pinecone ({index_name}).")
    return vector_store


# Example Usage
if __name__ == "__main__":
    json_file_path = "output-json/output-q3-2025.json"  # Ensure this file exists
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_content = f.read()

    vector_store = index_json_content(json_content, filename=json_file_path, index_name="json-index-1", region="us-east-1")
    if vector_store:
        print("✅ Indexing completed successfully.")
    else:
        print("❌ Indexing failed.")
