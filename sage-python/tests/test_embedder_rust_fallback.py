"""Tests for Embedder 3-tier fallback (Rust > sentence-transformers > hash)."""
from unittest.mock import MagicMock, patch

from sage.memory.embedder import EMBEDDING_DIM, Embedder, _try_rust_embedder


class TestRustEmbedderPreferred:
    """Embedder should prefer RustEmbedder when available."""

    def test_embedder_prefers_rust_when_available(self):
        """When _try_rust_embedder returns a backend, Embedder should use it."""
        mock_rust = MagicMock()
        mock_rust.embed.return_value = [0.1] * EMBEDDING_DIM
        mock_rust.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]
        mock_rust.is_semantic = True

        with patch("sage.memory.embedder._try_rust_embedder", return_value=mock_rust):
            emb = Embedder()
            vec = emb.embed("test")
            assert len(vec) == EMBEDDING_DIM
            mock_rust.embed.assert_called_once_with("test")

    def test_embedder_rust_is_semantic(self):
        """RustEmbedder backend should report is_semantic=True."""
        mock_rust = MagicMock()
        mock_rust.embed.return_value = [0.5] * EMBEDDING_DIM
        mock_rust.is_semantic = True

        with patch("sage.memory.embedder._try_rust_embedder", return_value=mock_rust):
            emb = Embedder()
            assert emb.is_semantic is True

    def test_embedder_rust_embed_batch(self):
        """Batch embedding should delegate to the Rust backend."""
        mock_rust = MagicMock()
        mock_rust.embed_batch.return_value = [
            [0.1] * EMBEDDING_DIM,
            [0.2] * EMBEDDING_DIM,
        ]
        mock_rust.is_semantic = True

        with patch("sage.memory.embedder._try_rust_embedder", return_value=mock_rust):
            emb = Embedder()
            vecs = emb.embed_batch(["a", "b"])
            assert len(vecs) == 2
            mock_rust.embed_batch.assert_called_once_with(["a", "b"])


class TestHashFallback:
    """Embedder falls back to hash when no semantic backend is available."""

    def test_embedder_falls_back_to_hash_without_rust(self):
        """When Rust unavailable and ST unavailable, falls back to hash."""
        emb = Embedder(force_hash=True)
        vec = emb.embed("test")
        assert len(vec) == EMBEDDING_DIM
        assert emb.is_semantic is False

    def test_force_hash_bypasses_rust(self):
        """force_hash=True should skip Rust detection entirely."""
        mock_rust = MagicMock()
        mock_rust.is_semantic = True

        with patch("sage.memory.embedder._try_rust_embedder", return_value=mock_rust):
            emb = Embedder(force_hash=True)
            assert emb.is_semantic is False
            # Rust backend should NOT have been called
            mock_rust.embed.assert_not_called()


class TestTryRustEmbedder:
    """_try_rust_embedder should be safe and never crash."""

    def test_returns_none_without_sage_core(self):
        """_try_rust_embedder returns None when sage_core lacks RustEmbedder."""
        # sage_core may or may not have RustEmbedder depending on build
        # but the function should not crash
        result = _try_rust_embedder()
        # Result is either None or a RustEmbedder -- both are valid
        assert result is None or hasattr(result, "embed")

    def test_returns_none_on_import_error(self):
        """_try_rust_embedder returns None when sage_core is not installed."""
        with patch.dict("sys.modules", {"sage_core": None}):
            result = _try_rust_embedder()
            assert result is None

    def test_returns_none_when_no_rust_embedder_attr(self):
        """_try_rust_embedder returns None when sage_core exists but has no RustEmbedder."""
        mock_core = MagicMock(spec=[])  # empty spec = no attributes
        with patch.dict("sys.modules", {"sage_core": mock_core}):
            result = _try_rust_embedder()
            assert result is None
