from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def load_and_chunk_file(file_path: str, embedding_model:str, openai_client: OpenAI) -> list:
    """Load a pdf file and split it into chunks by page."""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load_and_split()
    
    return [
        {
            "content": page.page_content,
            "source": page.metadata["source"],
            "page_number": page.metadata["page"],
            "embedding": openai_client.embeddings.create(
                input=[page.page_content], 
                model=embedding_model
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

def store_files_to_collection(
    collection_name: str, 
    files: list[str], 
    embedding_model:str, 
    client: QdrantClient, 
    openai_client: OpenAI
    ) -> None:
    """Load pdf files, chunk them, and save them to a qdrant collection."""
    
    for file_path in files:
        chunks = load_and_chunk_file(file_path, embedding_model, openai_client)
        save_chunks_to_collection(client, collection_name, chunks)
