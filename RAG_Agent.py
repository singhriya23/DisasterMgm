# langraph_rag_pipeline.py

import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document
from hybrid_search_pinecone_assign5 import query_pinecone_with_gpt  # your existing function

# ✅ Load environment variables from .env
load_dotenv(dotenv_path=".env")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise EnvironmentError("❌ PINECONE_API_KEY not found in .env file!")

# ----------------------------
# 1. Define LangGraph state
# ----------------------------

class RAGState(TypedDict, total=False):
    question: str
    year: Optional[int]
    quarter: Optional[str]
    top_k: Optional[int]
    rag_output: str

# ----------------------------
# 2. RAG Agent using metadata filter only
# ----------------------------

def rag_agent(state: RAGState) -> Dict[str, Any]:
    query = state.get("question", "Summarize NVIDIA’s performance.")
    top_k = state.get("top_k", 5)

    try:
        # 🔍 Perform hybrid search with GPT-4o answer generation
        response_text = query_pinecone_with_gpt(
            query=query,
            index_name="json-index-1",     # ✅ Match your Pinecone index
            region="us-east-1",
            top_k=top_k
        )

        return {"rag_output": response_text}

    except Exception as e:
        return {"rag_output": f"❌ RAG Agent error: {str(e)}"}


# ----------------------------
# 3. Build LangGraph pipeline
# ----------------------------

def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("RAGAgent", RunnableLambda(rag_agent))
    builder.set_entry_point("RAGAgent")
    builder.add_edge("RAGAgent", END)
    return builder.compile()

# ----------------------------
# 4. Example usage
# ----------------------------

if __name__ == "__main__":
    graph = build_graph()

    sample_state = {
        "question": "What are the risks associated with NVIDIA’s performance in Q1 2021?",
        "year": 2021,
        "quarter": "Q1",
        "top_k": 3
    }

    result = graph.invoke(sample_state)

    print("\n📝 RAG Agent Output:\n")
    print(result.get("rag_output"))
