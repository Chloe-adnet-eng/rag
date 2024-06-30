from openai import OpenAI
from qdrant_client import QdrantClient
from params import COLLECTION_NAME, OPENAI_KEY, TOP_VECTOR_SEARCH
import os

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

class RagAssistant:
    """A class to represent a RAG assistant."""

    def __init__(
        self,
        openai_client: OpenAI,
        qdrant_client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
        top_k_vectors: int = TOP_VECTOR_SEARCH,
        model: str = "gpt-3.5-turbo",
    ):
        """Initialize the assistant."""
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.top_k_vectors = top_k_vectors
        self.model = model

    def retrieve(self, query: str) -> list[str]:
        """Retrieve the top k vectors related to the given question from a Qdrant DB."""
        embedded_query = (
            self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-ada-002",
            )
            .data[0]
            .embedding
        )
        query_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedded_query,
            limit=self.top_k_vectors,
        )
        return [
            f"{vector.payload['content']}"
            for vector in query_results
            if vector.payload is not None
        ]

    def generate_answer(self, query: str, context_chunks: list[str]) -> str | None:
        """Generate an answer to a question using OpenAI API with added context base on a Qdrant DB."""
        context = "\n".join(context_chunks)
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant.
                    You will be provided with a document delimited by "Context:" and a question.
                    Your task is to answer the question using only the provided document.
                    Give the most relevant pieces of information in the document that answer the given question.
                    """,
                },
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"{query}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def query(self, query: str) -> str | None:
        """Query the assistant."""
        context_chunks = self.retrieve(query)
        answer = self.generate_answer(query, context_chunks)
        return answer

if __name__ == "__main__":
    qdrant_client = QdrantClient(
        host="localhost",
        prefer_grpc=True,
    )
    openai_client = OpenAI()
    rag = RagAssistant(openai_client, qdrant_client)

    while True:
        user_query = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if user_query.lower() == 'exit':
            print("Au revoir!")
            break
        answer = rag.query(user_query)
        print(f"Réponse: {answer}")
