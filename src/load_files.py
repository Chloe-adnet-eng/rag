from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from params import EMBEDDING_MODEL, OPENAI_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
openai_client = OpenAI()

def load_and_chunk_file(file_path: str):
    """Load a pdf file and split it into chunks by page."""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load_and_split()
    return [
        {
            "content": page.page_content,
            "source": page.metadata["source"],
            "page_number": page.metadata["page"],
            "embedding": openai_client.embeddings.create(
                input=[page.page_content], model=EMBEDDING_MODEL
            )
            .data[0]
            .embedding,
        }
        for page in pages
    ]

def save_chunks_to_collection(
    qdrant_client: QdrantClient, collection_name: str, chunks: list[dict[str, Any]]
) -> None:
    """Save chunks to a collection in Qdrant, payload contains metadata."""
    for chunk in chunks:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk["embedding"],
                    payload=chunk,
                )
            ],
        )

def store_files_to_collection(collection_name: str, files: list[str]):
    """Load pdf files, chunk them, and save them to a qdrant collection."""
    qdrant_client = QdrantClient(
        host="localhost",
        prefer_grpc=True,
    )
    for file_path in files:
        chunks = load_and_chunk_file(file_path)
        save_chunks_to_collection(qdrant_client, collection_name, chunks)

if __name__ == "__main__":
    store_files_to_collection("tuto_rag", ["cours_maths.pdf"])