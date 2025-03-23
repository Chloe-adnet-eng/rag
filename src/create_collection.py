from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def create_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create a collection in Qdrant."""

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size, 
                distance=Distance.COSINE),
        )
    except:
        print('La collection existe déjà!')

