from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

DATA_FOLDER = "data"

documents = []

for file in os.listdir(DATA_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_FOLDER, file)

        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                documents.append({
                    "text": text,
                    "source": file,
                    "page": page_num + 1
                })

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for doc in documents:

    split_text = splitter.split_text(doc["text"])

    for chunk in split_text:

        chunks.append({
            "text": chunk,
            "source": doc["source"],
            "page": doc["page"]
        })

print("Chunks:", len(chunks))

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

client = chromadb.PersistentClient(
    path="db"
)

collection = client.get_or_create_collection(
    name="documents"
)

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))],
    metadatas=[
        {
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]
)

print("Indexing Complete")