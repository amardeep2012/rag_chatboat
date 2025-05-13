from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

class Retriever:
    def __init__(self, index_path, embed_model="all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(embed_model)
        with open(index_path.replace(".index", "_docs.pkl"), "rb") as f:
            self.docs = pickle.load(f)

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            results.append({"text": self.docs[i], "score": score})
        return results
