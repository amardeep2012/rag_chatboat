import os
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_all_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            print(f"Loaded PDF: {filename} - {len(text)} chars")
            docs.append(text)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(file_path)
            print(f"Loaded DOCX: {filename} - {len(text)} chars")
            docs.append(text)
        elif filename.lower().endswith(".txt"):
            text = open(file_path, "r", encoding="utf-8").read()
            print(f"Loaded TXT: {filename} - {len(text)} chars")
            docs.append(text)
    return docs

# Load documents from data folder
docs = load_all_documents("data")

# Check document chunks
print(f"Total documents loaded: {len(docs)}")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding documents...")
embeddings = model.encode(docs)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and docs
output_dir = "embeddings/faiss_index"
os.makedirs(output_dir, exist_ok=True)
faiss.write_index(index, f"{output_dir}/index.index")
with open(f"{output_dir}/index_docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print(f"FAISS index and docs saved to {output_dir}")
