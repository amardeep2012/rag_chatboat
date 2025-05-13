from fastapi import FastAPI
from pydantic import BaseModel
from app.retriever import Retriever
from app.rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot for Customer Support",
    version="1.0"
)

# Initialize Retriever and Pipeline
retriever = Retriever("embeddings/faiss_index/index.index")
rag_pipeline = RAGPipeline(retriever)

# Define request schema
class QueryRequest(BaseModel):
    query: str

# Define response schema
class AnswerResponse(BaseModel):
    answer: str
    retrieved_sources: list

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QueryRequest):
    answer, retrieved = rag_pipeline.answer(req.query)
    sources = [{"text": doc['text'], "score": float(doc['score'])} for doc in retrieved]
    return AnswerResponse(answer=answer, retrieved_sources=sources)
