import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="db"
)

collection = client.get_collection(
    name="documents"
)


def ask_question(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    context = ""

    for doc in retrieved_docs:
        context += doc + "\n\n"

    prompt = f"""
Answer ONLY using the provided context.

If the answer is not present in the context,
respond with:

I cannot find the answer in the provided documents.

Context:
{context}

Question:
{question}
"""

    response = gemini_model.generate_content(
        prompt
    )
    if "I cannot find the answer" in response.text:
        return response.text, []
    sources = []

    shown = set()

    for meta in retrieved_meta:
        source = f"{meta['source']} | Page {meta['page']}"

        if source not in shown:
            sources.append(source)
            shown.add(source)

    return response.text, sources