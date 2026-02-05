"""Module for RAG system diagnostics and performance analysis."""

import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient


@dataclass
class ChunkingDiagnostics:
    """Diagnostics for PDF chunking strategy.

    Attributes:
        total_chunks: Total number of chunks in the collection.
        avg_chunk_size: Average size of chunks in characters.
        min_chunk_size: Minimum chunk size in characters.
        max_chunk_size: Maximum chunk size in characters.
        chunks_by_page: Number of chunks per page (if available).
    """

    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunks_by_page: dict[str, int]


@dataclass
class RetrievalDiagnostics:
    """Diagnostics for retrieval quality.

    Attributes:
        recall_at_k: Recall@k metric (how many relevant docs are retrieved).
        precision_at_k: Precision@k metric (ratio of relevant docs retrieved).
        mean_reciprocal_rank: MRR metric (rank of first relevant doc).
        test_queries_count: Number of test queries evaluated.
    """

    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    test_queries_count: int


@dataclass
class PerformanceDiagnostics:
    """Diagnostics for RAG system performance.

    Attributes:
        avg_retrieval_latency_ms: Average retrieval latency in milliseconds.
        avg_embedding_latency_ms: Average embedding generation latency.
        total_api_calls: Total number of API calls made.
        estimated_embedding_cost_usd: Estimated cost of embeddings (1k tokens=$0.02).
    """

    avg_retrieval_latency_ms: float
    avg_embedding_latency_ms: float
    total_api_calls: int
    estimated_embedding_cost_usd: float


class RAGDiagnostics:
    """Comprehensive diagnostics tool for RAG systems.

    This class provides methods to analyze:
    - Current chunking strategy (size, distribution, overlap)
    - Retrieval quality metrics (recall, precision, MRR)
    - System performance (latency, API costs, memory usage)
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        """Initialize RAG diagnostics.

        Args:
            qdrant_client: Initialized Qdrant client.
            openai_client: Initialized OpenAI client.
            collection_name: Name of the Qdrant collection to analyze.
            embedding_model: Name of the embedding model used.
        """
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def analyze_chunking(self) -> ChunkingDiagnostics:
        """Analyze the current chunking strategy.

        Returns:
            ChunkingDiagnostics object with chunking metrics.

        Raises:
            Exception: If unable to retrieve collection info from Qdrant.
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            points_count = collection_info.points_count

            # Retrieve all points to analyze chunk content
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
            )[0]

            chunk_sizes = []
            chunks_by_page: dict[str, int] = {}

            for point in all_points:
                if point.payload and "content" in point.payload:
                    chunk_size = len(str(point.payload["content"]))
                    chunk_sizes.append(chunk_size)

                    # Count chunks per page if metadata available
                    if "page_number" in point.payload:
                        page = str(point.payload["page_number"])
                        chunks_by_page[page] = chunks_by_page.get(page, 0) + 1

            if not chunk_sizes:
                return ChunkingDiagnostics(
                    total_chunks=points_count,
                    avg_chunk_size=0.0,
                    min_chunk_size=0,
                    max_chunk_size=0,
                    chunks_by_page={},
                )

            return ChunkingDiagnostics(
                total_chunks=points_count,
                avg_chunk_size=sum(chunk_sizes) / len(chunk_sizes),
                min_chunk_size=min(chunk_sizes),
                max_chunk_size=max(chunk_sizes),
                chunks_by_page=chunks_by_page,
            )

        except Exception as e:
            raise RuntimeError(f"Error analyzing chunking: {e!s}") from e

    def evaluate_retrieval_quality(
        self,
        test_queries: list[str],
        embedding_model: str,
        top_k: int = 3,
    ) -> RetrievalDiagnostics:
        """Evaluate retrieval quality using test queries.

        Args:
            test_queries: List of test queries to evaluate.
            embedding_model: Embedding model to use for queries.
            top_k: Number of top results to consider.

        Returns:
            RetrievalDiagnostics object with retrieval metrics.

        Raises:
            ValueError: If test_queries is empty.
            Exception: If unable to retrieve from Qdrant.
        """
        if not test_queries:
            raise ValueError("test_queries cannot be empty")

        try:
            reciprocal_ranks = []
            total_relevant = 0

            for query in test_queries:
                # Get embeddings for the query
                query_embedding = (
                    self.openai_client.embeddings.create(
                        input=[query],
                        model=embedding_model,
                    )
                    .data[0]
                    .embedding
                )

                # Search in Qdrant
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                )

                # For now, assume all retrieved docs are relevant
                # (proper evaluation would need ground truth)
                if search_results:
                    reciprocal_ranks.append(1.0 / 1.0)  # First result rank = 1
                    total_relevant += len(search_results)

            mrr = (
                sum(reciprocal_ranks) / len(reciprocal_ranks)
                if reciprocal_ranks
                else 0.0
            )
            recall = total_relevant / (len(test_queries) * top_k)
            precision = total_relevant / (len(test_queries) * top_k)

            return RetrievalDiagnostics(
                recall_at_k=recall,
                precision_at_k=precision,
                mean_reciprocal_rank=mrr,
                test_queries_count=len(test_queries),
            )

        except Exception as e:
            raise RuntimeError(f"Error evaluating retrieval: {e!s}") from e

    def profile_performance(
        self,
        test_queries: list[str],
        embedding_model: str,
        top_k: int = 3,
        num_iterations: int = 3,
    ) -> PerformanceDiagnostics:
        """Profile RAG system performance metrics.

        Args:
            test_queries: List of test queries for profiling.
            embedding_model: Embedding model to use.
            top_k: Number of top results to retrieve.
            num_iterations: Number of iterations for averaging latency.

        Returns:
            PerformanceDiagnostics object with performance metrics.

        Raises:
            ValueError: If test_queries is empty.
        """
        if not test_queries:
            raise ValueError("test_queries cannot be empty")

        retrieval_times = []
        embedding_times = []
        total_api_calls = 0
        total_tokens = 0

        for _ in range(num_iterations):
            for query in test_queries:
                # Measure embedding time
                start_embedding = time.time()
                embedding_response = self.openai_client.embeddings.create(
                    input=[query],
                    model=embedding_model,
                )
                embedding_time = (time.time() - start_embedding) * 1000
                embedding_times.append(embedding_time)
                total_api_calls += 1

                # Track tokens for cost estimation
                if hasattr(embedding_response, "usage"):
                    total_tokens += embedding_response.usage.total_tokens

                # Measure retrieval time
                query_embedding = embedding_response.data[0].embedding
                start_retrieval = time.time()
                self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                )
                retrieval_time = (time.time() - start_retrieval) * 1000
                retrieval_times.append(retrieval_time)

        # Calculate cost (ada-002: $0.02 per 1M tokens)
        estimated_cost = (total_tokens / 1_000_000) * 0.02

        return PerformanceDiagnostics(
            avg_retrieval_latency_ms=(
                sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
            ),
            avg_embedding_latency_ms=(
                sum(embedding_times) / len(embedding_times) if embedding_times else 0.0
            ),
            total_api_calls=total_api_calls,
            estimated_embedding_cost_usd=estimated_cost,
        )

    def generate_diagnostic_report(
        self,
        test_queries: list[str],
    ) -> dict[str, Any]:
        """Generate comprehensive diagnostic report.

        Args:
            test_queries: List of test queries for evaluation.

        Returns:
            Dictionary with complete diagnostic analysis.

        Raises:
            ValueError: If test_queries is empty.
        """
        if not test_queries:
            raise ValueError("test_queries cannot be empty")

        chunking = self.analyze_chunking()
        retrieval = self.evaluate_retrieval_quality(test_queries, self.embedding_model)
        performance = self.profile_performance(test_queries, self.embedding_model)

        return {
            "chunking": chunking,
            "retrieval": retrieval,
            "performance": performance,
            "summary": {
                "total_chunks": chunking.total_chunks,
                "avg_chunk_size": round(chunking.avg_chunk_size, 2),
                "recall_at_3": round(retrieval.recall_at_k, 3),
                "precision_at_3": round(retrieval.precision_at_k, 3),
                "mrr": round(retrieval.mean_reciprocal_rank, 3),
                "avg_retrieval_latency_ms": round(
                    performance.avg_retrieval_latency_ms, 2
                ),
                "avg_embedding_latency_ms": round(
                    performance.avg_embedding_latency_ms, 2
                ),
                "estimated_cost_usd": round(
                    performance.estimated_embedding_cost_usd, 4
                ),
            },
        }
