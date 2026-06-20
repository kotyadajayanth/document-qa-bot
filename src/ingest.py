from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

DATA_FOLDER = "data"

all_documents = []

for file in os.listdir(DATA_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_FOLDER, file)

        reader = PdfReader(pdf_path)

        print(f"\nReading: {file}")

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                all_documents.append({
                    "text": text,
                    "source": file,
                    "page": page_num + 1
                })

print("\nTotal Pages Loaded:", len(all_documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for doc in all_documents:

    split_text = splitter.split_text(doc["text"])

    for chunk in split_text:

        chunks.append({
            "text": chunk,
            "source": doc["source"],
            "page": doc["page"]
        })

print("Total Chunks Created:", len(chunks))

print("\nSample Chunk:\n")
print(chunks[0]["text"][:500])