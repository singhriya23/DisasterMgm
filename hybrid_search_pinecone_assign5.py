import os
import re
import openai
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv  # Load environment variables

# ✅ Load .env file
load_dotenv(dotenv_path=".env")

# ✅ Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # GPT-4o
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Pinecone

# ✅ Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Function to Extract Quarter from Query
def extract_quarter(query):
    """Extracts quarter and year from the user query using regex."""
    match = re.search(r"(Q[1-4])\s*(\d{4})", query, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2)  # Returns ("Q3", "2023")
    return None, None

# ✅ Hybrid Search Function for GPT-4o
def query_pinecone_with_gpt(query, index_name="json-index-1", region="us-east-1", top_k=5):
    """ Query Pinecone with hybrid search (semantic + keyword-based) and generate an answer using GPT-4o. """

    quarter, year = extract_quarter(query)

    # ✅ Initialize Pinecone Client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # ✅ Load Pinecone index
    index = pc.Index(index_name)

    # ✅ Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Initialize Vector Store
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="page_content")

    # ✅ Initialize Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # ✅ Perform Hybrid Search
    semantic_results = retriever.get_relevant_documents(query)
    keyword_results = vector_store.similarity_search(query, k=top_k)

    # ✅ Merge & Weight Results (Semantic: 70%, Keyword: 30%)
    ranked_results = [(doc.page_content, 0.7) for doc in semantic_results] + \
                     [(doc.page_content, 0.3) for doc in keyword_results]

    # ✅ Sort & Deduplicate Results
    unique_results = {}
    for content, score in sorted(ranked_results, key=lambda x: x[1], reverse=True):
        unique_results[content] = score  # Ensures highest score is kept

    # ✅ Extract Final Sorted List
    final_results = list(unique_results.keys())[:top_k]

    # ✅ Debugging: Print Retrieved Chunks
    print("🔍 Retrieved Chunks (before filtering):", final_results)

    # ✅ Quarter-Year Filtering (Fix: Don't remove all results!)
    if quarter and year:
        pattern = re.compile(rf"{quarter}.*{year}", re.IGNORECASE)
        priority_results = [doc for doc in final_results if pattern.search(doc)]
        final_results = priority_results if priority_results else final_results  # Avoid empty list

    # ✅ Final Debugging
    print("🔍 Final Chunks (after filtering):", final_results)

    if not final_results:
        return f"I couldn't find relevant information for {quarter} {year}."

    # ✅ Prepare context for GPT-4o
    top_chunks = "\n\n".join(final_results[:3])

    # ✅ Generate answer using GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI financial assistant that answers questions based on reports."},
            {"role": "user", "content": f"Context:\n{top_chunks}\n\nQuestion: {query}\nAnswer based on the above context:"}
        ]
    )

    return response.choices[0].message.content


