import streamlit as st
from app.retriever import Retriever
from app.rag_pipeline import RAGPipeline

retriever = Retriever("embeddings/faiss_index/index.index")
rag_pipeline = RAGPipeline(retriever)

st.title("RAG Customer Support Chatbot")
query = st.text_input("Ask your question:")

if query:
    answer, retrieved = rag_pipeline.answer(query)
    st.write(f"**Answer:** {answer}")
    with st.expander("See retrieved documents"):
        for doc in retrieved:
            st.write(f"- {doc['text'][:300]}... (Score: {doc['score']})")
