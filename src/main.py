from qdrant_client import QdrantClient
from openai import OpenAI
from create_collection import create_collection
from load_files import store_files_to_collection
from params import get_variables
from rag import start_rag
import os

COLLECTION_NAME='math_report'
OPEN_AI_EMBEDDINGS_SIZE = 1536
EMBEDDING_MODEL='text-embedding-ada-002'
TOP_K_VECTORS=3

if __name__ == '__main__':
    
    variables = get_variables()
    os.environ["OPENAI_API_KEY"] = variables.OPENAI_API_KEY
    
    qdrant_client = QdrantClient(
            host="localhost",
        )

    openai_client = OpenAI()
    
    create_collection(
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME, 
        vector_size=OPEN_AI_EMBEDDINGS_SIZE
    )
    
    
    store_files_to_collection(
        collection_name=COLLECTION_NAME, 
        files=["cours_maths.pdf"], 
        embedding_model=EMBEDDING_MODEL,
        client=qdrant_client, 
        openai_client=openai_client
    )
    
    start_rag(
        openai_client=openai_client,
        embedding_model=EMBEDDING_MODEL,
        qdrant_client=qdrant_client, 
        collection_name=COLLECTION_NAME, 
        top_k_vectors=TOP_K_VECTORS
    )
