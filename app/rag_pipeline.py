from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
url = os.getenv("URL")
class RAGPipeline:
    def __init__(self, retriever, threshold=0.7):
        self.retriever = retriever
        self.threshold = threshold

    def answer(self, query):
        retrieved = self.retriever.retrieve(query)
        if not retrieved or retrieved[0]['score'] < self.threshold:
            return "I Don't know", []
        context = "\n".join([doc['text'] for doc in retrieved])
        prompt = (
            "Answer the following question using ONLY the provided context. "
            "If the answer is not in the context, reply exactly with: I Don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

        client = OpenAI(
            base_url=url,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        # If the model says it doesn't know, standardize the response
        if "no information" in answer.lower() or "i don't know" in answer.lower():
            return "I Don't know", retrieved
        return answer, retrieved
