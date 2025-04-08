import os
import json
from pinecone import Pinecone, ServerlessSpec  
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv  # ✅ Import dotenv

# ✅ Load .env once at the top
load_dotenv(dotenv_path=".env")


def index_json_content(json_content, index_name="json-index", region="us-east-1"):
    """
    Index a JSON content (as string) into Pinecone after chunking.

    Args:
        json_content (str): The JSON content as a string.
        index_name (str): Name of the Pinecone index (must be lowercase, alphanumeric, and use '-' instead of '_').
        region (str, optional): Pinecone region (default: us-east-1).

    Returns:
        PineconeVectorStore: The indexed Pinecone vector store.
    """
    
    # Ensure index name follows Pinecone rules
    index_name = index_name.lower().replace("_", "-")  # Convert to lowercase & replace underscores

    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        raise ValueError("❌ Pinecone API Key is missing! Please check your .env file.")

    # Initialize Pinecone Client
    pc = Pinecone(api_key=pinecone_api_key)

    # Retrieve list of existing indexes
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    # If the index does not exist, create it
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating it now...")

        pc.create_index(
            name=index_name,  
            dimension=384,  # Ensure dimension matches the embedding model
            metric="cosine",  
            spec=ServerlessSpec(cloud="aws", region=region)  
        )

    # Retrieve the correct Pinecone Index object
    index = pc.Index(index_name)

    # Load Embeddings
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Initialize the Vector Store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="page_content",
    )

    # Try parsing the JSON content
    try:
        if isinstance(json_content, str):  # If the content is a string, parse it
            data = json.loads(json_content)  # Convert string content to JSON (dictionary)
        else:
            data = json_content  # If it's already a dict, use it directly
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format in the content.")

    # Extract text chunks safely
    if 'chunks' not in data or not isinstance(data['chunks'], list):
        raise ValueError("❌ No valid chunks found in the JSON content.")

    # Ensure each chunk is extracted properly
    chunks = [chunk if isinstance(chunk, str) else chunk.get("content", "") for chunk in data["chunks"]]

    if not chunks:
        raise ValueError("❌ No content found in the JSON chunks.")

    # Convert chunks to LangChain Documents
    documents = [Document(page_content=chunk, metadata={"source": "in-memory"}) for chunk in chunks if chunk]

    # Insert chunks into Pinecone
    if documents:
        vector_store.add_documents(documents)
        print(f"✅ Successfully indexed {len(documents)} chunks into Pinecone ({index_name}).")
    else:
        print("⚠️ No chunks were created. Check if the JSON content contains text.")

    return vector_store
