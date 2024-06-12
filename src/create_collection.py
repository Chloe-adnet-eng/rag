from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from params import COLLECTION_NAME, OPEN_AI_EMBEDDINGS_SIZE


def main(collection_name: str = COLLECTION_NAME, vector_size: int = OPEN_AI_EMBEDDINGS_SIZE):
    """Create a collection in Qdrant."""
    qdrant_client = QdrantClient(
            host="localhost",
            prefer_grpc=True,
        )

    qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

if __name__ == "__main__":
    main()
