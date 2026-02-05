"""Tests for RAG diagnostics module."""

from unittest.mock import MagicMock, Mock

import pytest
from qdrant_client import QdrantClient

from src.diagnostics import (
    ChunkingDiagnostics,
    PerformanceDiagnostics,
    RAGDiagnostics,
    RetrievalDiagnostics,
)


@pytest.fixture()
def mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client."""
    return MagicMock(spec=QdrantClient)


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    client = MagicMock()
    # Mock embeddings.create method
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    mock_embedding_response.usage = Mock(total_tokens=10)
    client.embeddings.create.return_value = mock_embedding_response
    return client


@pytest.fixture()
def diagnostics(
    mock_qdrant_client: MagicMock,
    mock_openai_client: MagicMock,
) -> RAGDiagnostics:
    """Create a RAGDiagnostics instance with mocked clients."""
    return RAGDiagnostics(
        qdrant_client=mock_qdrant_client,
        openai_client=mock_openai_client,
        collection_name="test_collection",
        embedding_model="text-embedding-ada-002",
    )


class TestChunkingDiagnostics:
    """Tests for ChunkingDiagnostics dataclass."""

    def test_chunking_diagnostics_creation(self) -> None:
        """Test creation of ChunkingDiagnostics object."""
        diag = ChunkingDiagnostics(
            total_chunks=10,
            avg_chunk_size=500.0,
            min_chunk_size=100,
            max_chunk_size=1000,
            chunks_by_page={"1": 5, "2": 5},
        )

        assert diag.total_chunks == 10
        assert diag.avg_chunk_size == 500.0
        assert diag.min_chunk_size == 100
        assert diag.max_chunk_size == 1000
        assert diag.chunks_by_page == {"1": 5, "2": 5}


class TestRetrievalDiagnostics:
    """Tests for RetrievalDiagnostics dataclass."""

    def test_retrieval_diagnostics_creation(self) -> None:
        """Test creation of RetrievalDiagnostics object."""
        diag = RetrievalDiagnostics(
            recall_at_k=0.8,
            precision_at_k=0.75,
            mean_reciprocal_rank=0.9,
            test_queries_count=10,
        )

        assert diag.recall_at_k == 0.8
        assert diag.precision_at_k == 0.75
        assert diag.mean_reciprocal_rank == 0.9
        assert diag.test_queries_count == 10


class TestPerformanceDiagnostics:
    """Tests for PerformanceDiagnostics dataclass."""

    def test_performance_diagnostics_creation(self) -> None:
        """Test creation of PerformanceDiagnostics object."""
        diag = PerformanceDiagnostics(
            avg_retrieval_latency_ms=50.0,
            avg_embedding_latency_ms=100.0,
            total_api_calls=30,
            estimated_embedding_cost_usd=0.0006,
        )

        assert diag.avg_retrieval_latency_ms == 50.0
        assert diag.avg_embedding_latency_ms == 100.0
        assert diag.total_api_calls == 30
        assert diag.estimated_embedding_cost_usd == 0.0006


class TestAnalyzeChunking:
    """Tests for analyze_chunking method."""

    def test_analyze_chunking_success(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful chunking analysis."""
        # Mock collection info
        mock_collection = Mock()
        mock_collection.points_count = 10
        mock_qdrant_client.get_collection.return_value = mock_collection

        # Mock points
        mock_points = [
            Mock(
                payload={
                    "content": "x" * 500,
                    "page_number": 1,
                }
            ),
            Mock(
                payload={
                    "content": "y" * 700,
                    "page_number": 1,
                }
            ),
            Mock(
                payload={
                    "content": "z" * 300,
                    "page_number": 2,
                }
            ),
        ]
        mock_qdrant_client.scroll.return_value = (mock_points, None)

        result = diagnostics.analyze_chunking()

        assert result.total_chunks == 10
        assert result.avg_chunk_size == pytest.approx(500.0, rel=1e-2)
        assert result.min_chunk_size == 300
        assert result.max_chunk_size == 700
        assert result.chunks_by_page == {"1": 2, "2": 1}

    def test_analyze_chunking_empty_collection(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test chunking analysis with empty collection."""
        mock_collection = Mock()
        mock_collection.points_count = 0
        mock_qdrant_client.get_collection.return_value = mock_collection
        mock_qdrant_client.scroll.return_value = ([], None)

        result = diagnostics.analyze_chunking()

        assert result.total_chunks == 0
        assert result.avg_chunk_size == 0.0
        assert result.min_chunk_size == 0
        assert result.max_chunk_size == 0

    def test_analyze_chunking_error(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test chunking analysis error handling."""
        mock_qdrant_client.get_collection.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Error analyzing chunking"):
            diagnostics.analyze_chunking()


class TestEvaluateRetrievalQuality:
    """Tests for evaluate_retrieval_quality method."""

    def test_evaluate_retrieval_success(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test successful retrieval quality evaluation."""
        test_queries = ["What is calculus?", "Explain derivatives"]

        # Mock search results
        mock_result = Mock()
        mock_result.payload = {"content": "Some content"}
        mock_qdrant_client.search.return_value = [mock_result]

        result = diagnostics.evaluate_retrieval_quality(
            test_queries, "text-embedding-ada-002"
        )

        assert result.test_queries_count == 2
        assert result.recall_at_k > 0
        assert result.precision_at_k > 0
        assert result.mean_reciprocal_rank > 0

    def test_evaluate_retrieval_empty_queries(
        self,
        diagnostics: RAGDiagnostics,
    ) -> None:
        """Test retrieval evaluation with empty queries."""
        with pytest.raises(ValueError, match="test_queries cannot be empty"):
            diagnostics.evaluate_retrieval_quality([], "text-embedding-ada-002")

    def test_evaluate_retrieval_no_results(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test retrieval evaluation with no search results."""
        test_queries = ["Unknown query"]

        mock_qdrant_client.search.return_value = []

        result = diagnostics.evaluate_retrieval_quality(
            test_queries, "text-embedding-ada-002"
        )

        assert result.test_queries_count == 1
        assert result.recall_at_k == 0.0


class TestProfilePerformance:
    """Tests for profile_performance method."""

    def test_profile_performance_success(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test successful performance profiling."""
        test_queries = ["What is calculus?"]

        mock_qdrant_client.search.return_value = []

        result = diagnostics.profile_performance(
            test_queries, "text-embedding-ada-002", num_iterations=1
        )

        assert result.total_api_calls == 1
        assert result.avg_embedding_latency_ms > 0
        assert result.avg_retrieval_latency_ms >= 0
        assert result.estimated_embedding_cost_usd > 0

    def test_profile_performance_empty_queries(
        self,
        diagnostics: RAGDiagnostics,
    ) -> None:
        """Test performance profiling with empty queries."""
        with pytest.raises(ValueError, match="test_queries cannot be empty"):
            diagnostics.profile_performance([], "text-embedding-ada-002")

    def test_profile_performance_multiple_iterations(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test performance profiling with multiple iterations."""
        test_queries = ["Query 1", "Query 2"]

        mock_qdrant_client.search.return_value = []

        result = diagnostics.profile_performance(
            test_queries, "text-embedding-ada-002", num_iterations=2
        )

        # 2 queries * 2 iterations = 4 API calls
        assert result.total_api_calls == 4


class TestGenerateDiagnosticReport:
    """Tests for generate_diagnostic_report method."""

    def test_generate_diagnostic_report_success(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test successful diagnostic report generation."""
        test_queries = ["What is calculus?"]

        # Mock chunking analysis
        mock_collection = Mock()
        mock_collection.points_count = 5
        mock_qdrant_client.get_collection.return_value = mock_collection

        mock_points = [
            Mock(payload={"content": "x" * 500, "page_number": 1}),
        ]
        mock_qdrant_client.scroll.return_value = (mock_points, None)

        # Mock search
        mock_qdrant_client.search.return_value = []

        report = diagnostics.generate_diagnostic_report(test_queries)

        assert "chunking" in report
        assert "retrieval" in report
        assert "performance" in report
        assert "summary" in report
        assert report["summary"]["total_chunks"] == 5
        assert report["summary"]["recall_at_3"] >= 0
        assert report["summary"]["estimated_cost_usd"] >= 0

    def test_generate_diagnostic_report_empty_queries(
        self,
        diagnostics: RAGDiagnostics,
    ) -> None:
        """Test diagnostic report with empty queries."""
        with pytest.raises(ValueError, match="test_queries cannot be empty"):
            diagnostics.generate_diagnostic_report([])

    def test_generate_diagnostic_report_structure(
        self,
        diagnostics: RAGDiagnostics,
        mock_qdrant_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test diagnostic report structure."""
        test_queries = ["Test"]

        mock_collection = Mock()
        mock_collection.points_count = 1
        mock_qdrant_client.get_collection.return_value = mock_collection
        mock_qdrant_client.scroll.return_value = ([], None)
        mock_qdrant_client.search.return_value = []

        report = diagnostics.generate_diagnostic_report(test_queries)

        # Verify structure
        assert isinstance(report["chunking"], ChunkingDiagnostics)
        assert isinstance(report["retrieval"], RetrievalDiagnostics)
        assert isinstance(report["performance"], PerformanceDiagnostics)
        assert isinstance(report["summary"], dict)

        # Verify summary keys
        summary_keys = {
            "total_chunks",
            "avg_chunk_size",
            "recall_at_3",
            "precision_at_3",
            "mrr",
            "avg_retrieval_latency_ms",
            "avg_embedding_latency_ms",
            "estimated_cost_usd",
        }
        assert summary_keys == set(report["summary"].keys())
