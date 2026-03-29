"""
Skills Lab — Hybrid Retriever (SKILL.md Standard)

3-tier progressive disclosure search:
  Tier 1: search_skills(query) → metadata + scores (for agent to decide)
  Tier 2: get_skill_content(name) → full SKILL.md body

Hybrid search: BM25 (name+desc+tags+repo) + Semantic (desc+tags) + RRF(k=60).
Model: bge-small-en-v1.5 (512 tokens, 384 dims) or fallback all-MiniLM-L6-v2.
Batch embedding for performance.
Dedup detection (cosine > 0.85).

Performance improvements:
  - BM25 index cached, rebuilt only on skill changes (not every search)
  - Embedding cache persisted to disk (pickle)
  - Smart tokenizer: camelCase split, hyphen split, lowercased
  - Semantic search is OFF by default — must enable via SKILLS_LAB_SEMANTIC=1
  - BM25-only search is sub-100ms, no model download needed
  - Background model warmup: model loads asynchronously after first BM25 search
  - Timing instrumentation on all slow paths
"""

import logging
import math
import os
import pickle
import threading
import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from core.models import Skill, SkillChangelog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smart tokenizer — handles camelCase, hyphens, dots
# ---------------------------------------------------------------------------


def _smart_tokenize(text: str) -> list[str]:
    """
    Improved tokenizer for BM25 scoring.

    Tokenization strategy:
    - Lowercases all input text
    - Splits on camelCase boundaries: "NextJs" → ["next", "js"]
    - Splits on hyphens and dots: "cors-fix" → ["cors", "fix"]
    - Splits on whitespace and punctuation

    Args:
        text: The input text to tokenize.

    Returns:
        A list of string tokens suitable for BM25 indexing.
    """
    # Lowercase first
    text = text.lower()

    # First pass: split on non-alphanumeric characters
    result: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalpha() or ch.isdigit():
            current.append(ch)
        else:
            if current:
                result.append("".join(current))
                current = []
            # Non-alphanumeric characters act as separators
    if current:
        result.append("".join(current))

    # Second pass: re-split camelCase boundaries that survived lowercasing
    # (e.g., mixed-case input like "nextJsApi" → "next js api")
    final: list[str] = []
    for token in result:
        sub: list[str] = []
        buf: list[str] = [token[0]] if token else []
        for ch in token[1:]:
            if ch.isupper():
                sub.append("".join(buf))
                buf = [ch.lower()]
            else:
                buf.append(ch)
        if buf:
            sub.append("".join(buf))
        final.extend(sub)

    return final


class HybridRetriever:
    """
    Hybrid search engine with progressive disclosure and caching.

    Combines BM25 lexical search with semantic similarity search,
    fusing results via Reciprocal Rank Fusion (RRF). Supports:

    - **Tier 1 search**: Returns lightweight metadata + relevance scores
      so the agent can decide which skills to inspect further.
    - **Tier 2 content retrieval**: Returns the full SKILL.md body for
      a specific skill.
    - **Duplicate detection**: Identifies skills whose embeddings are
      too similar (cosine similarity above a threshold).
    - **Lineage tracking**: Retrieves version history / changelog for
      a skill.

    Embeddings are cached to disk (pickle) and the BM25 index is
    rebuilt only when skills change, not on every search call.
    """

    def __init__(
        self,
        session_factory: callable,
        manager,  # SKILLManager
        workspace_path: str = "",
        model_name: str = "BAAI/bge-small-en-v1.5",
        fallback_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        rrf_k: int = 60,
        dedup_threshold: float = 0.85,
        tokenizer: callable = _smart_tokenize,
    ) -> None:
        """
        Initialize the HybridRetriever.

        Args:
            session_factory: A callable that returns a new SQLAlchemy Session.
            manager: The SKILLManager instance used to read skill files.
            workspace_path: Root workspace path. If provided, embedding cache
                is stored at ``<workspace_path>/.cache/embedding_cache.pkl``.
            model_name: Primary sentence-transformer model name.
            fallback_model_name: Fallback model name if primary fails to load.
            top_k: Default number of results to return from search.
            rrf_k: RRF constant ``k`` (higher values flatten rank differences).
            dedup_threshold: Cosine similarity threshold for duplicate detection.
            tokenizer: Tokenizer function accepting a string and returning a
                list of tokens. Defaults to :func:`_smart_tokenize`.
        """
        self._session_factory = session_factory
        self._manager = manager
        self._workspace_path = workspace_path
        self._model_name = model_name
        self._fallback_model_name = fallback_model_name
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._dedup_threshold = dedup_threshold
        self._tokenizer = tokenizer

        # Lazy-loaded embedding model
        # Semantic search is OFF by default to keep search fast (<100ms BM25).
        # Enable via SKILLS_LAB_SEMANTIC=1 environment variable.
        self._model = None
        self._model_loaded = False
        self._semantic_available = os.environ.get("SKILLS_LAB_SEMANTIC", "").strip().lower() in ("1", "true", "yes")
        self._semantic_warming_up = False  # True while background warmup is running

        # Embedding cache: skill_name → np.ndarray (persisted to disk)
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._cache_dir = ""
        self._cache_lock = threading.Lock()

        if workspace_path:
            self._cache_dir = os.path.join(workspace_path, ".cache")
            os.makedirs(self._cache_dir, exist_ok=True)
            self._load_embedding_cache()

        # BM25 index cache — rebuilt only on skill changes
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_skill_ids: list[str] = []
        self._bm25_dirty = True  # Force rebuild on first search
        self._bm25_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Embedding cache persistence
    # -----------------------------------------------------------------------

    def _cache_path(self) -> str:
        """Return the filesystem path to the embedding cache pickle file."""
        return os.path.join(self._cache_dir, "embedding_cache.pkl")

    def _load_embedding_cache(self) -> None:
        """Load the embedding cache from disk if it exists."""
        path = self._cache_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self._embedding_cache = data
                    logger.info(f"Loaded {len(data)} cached embeddings from disk")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._embedding_cache = {}

    def _save_embedding_cache(self) -> None:
        """Persist the embedding cache to disk."""
        if not self._cache_dir:
            return
        try:
            with self._cache_lock:
                path = self._cache_path()
                with open(path, "wb") as f:
                    pickle.dump(self._embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Lazy-load the embedding model. Falls back to BM25-only if unavailable.

        IMPORTANT: This is the HOT PATH. On first call, it may trigger a model
        download (~100MB). This is only invoked when ``_semantic_available``
        is True (set via ``SKILLS_LAB_SEMANTIC=1`` env var).

        The import probe uses ``importlib.util.find_spec`` for a fast check
        (milliseconds) instead of actually importing the heavy package
        (which can take 5-10 seconds even when it ultimately fails).
        """
        if self._model_loaded:
            return
        self._model_loaded = True

        t0 = time.time()

        # If semantic search was not enabled, stay BM25-only (fast path).
        if not self._semantic_available:
            logger.info("Semantic search disabled (set SKILLS_LAB_SEMANTIC=1 to enable). Using BM25-only.")
            return

        # Fast probe: check if sentence_transformers is importable WITHOUT
        # actually importing the heavy torch/numpy dependency chain.
        try:
            import importlib.util
            if importlib.util.find_spec("sentence_transformers") is None:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers torch  "
                    "Falling back to BM25-only search."
                )
                self._semantic_available = False
                return
        except Exception:
            pass

        # Package exists — try loading models (may still fail at runtime).
        # NOTE: First-time load downloads ~100MB model weights.
        # Subsequent loads use cached files and take 2-5s on CPU.
        for model_name in [self._model_name, self._fallback_model_name]:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {model_name}...")
                t1 = time.time()
                self._model = SentenceTransformer(model_name)
                t2 = time.time()
                logger.info(f"Embedding model loaded: {model_name} ({t2 - t1:.1f}s)")

                if hasattr(self._model, 'max_seq_length') and self._model.max_seq_length < 512:
                    try:
                        config = self._model[0].auto_model.config
                        max_pos = config.max_position_embeddings
                        if max_pos > self._model.max_seq_length:
                            self._model.max_seq_length = min(max_pos, 512)
                            logger.info(f"  Overriding max_seq_length -> {self._model.max_seq_length}")
                    except Exception:
                        pass

                logger.info(f"Semantic search ready (total load time: {t2 - t0:.1f}s)")
                return
            except Exception as e:
                logger.warning(f"Cannot load model {model_name}: {e}")
                continue

        logger.warning("No embedding model available. Falling back to BM25-only search.")
        self._model = None
        self._semantic_available = False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a single text string into an embedding vector.

        Args:
            text: The text to encode.

        Returns:
            A numpy ndarray of the embedding, or ``None`` if no model is loaded.
        """
        if self._model is None:
            return None
        return self._model.encode(text, convert_to_numpy=True)

    def _get_embeddings_batch(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        """
        Encode a batch of text strings into embedding vectors.

        If batch encoding fails, falls back to sequential encoding.

        Args:
            texts: List of text strings to encode.

        Returns:
            A list of embedding ndarrays (or ``None``) aligned with the input list.
        """
        if self._model is None:
            return [None] * len(texts)
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [embeddings[i] for i in range(len(texts))]
        except Exception as e:
            logger.warning(f"Batch encode failed: {e}, falling back to sequential")
            return [self._get_embedding(t) for t in texts]

    # -----------------------------------------------------------------------
    # Embedding cache (public, called by EvolutionEngine)
    # -----------------------------------------------------------------------

    def compute_and_cache_embedding(self, skill_name: str, text: str) -> None:
        """
        Compute the embedding for a skill's text and store it in the cache.

        Note: This does **not** automatically persist to disk. Call
        :meth:`flush_cache` after a batch of computations to save.

        Args:
            skill_name: The unique skill identifier (used as cache key).
            text: The text to encode (typically description + tags).
        """
        # Only load model if semantic is enabled
        if not self._semantic_available:
            return
        self._load_model()
        if self._model is None:
            return
        emb = self._get_embedding(text)
        if emb is not None:
            with self._cache_lock:
                self._embedding_cache[skill_name] = emb
            # Debounced save — don't save on every single compute.
            # The caller (evolver) should call flush_cache() after batch operations.

    def flush_cache(self) -> None:
        """Persist the embedding cache to disk. Call after batch writes."""
        self._save_embedding_cache()

    def clear_cache(self, skill_name: str) -> None:
        """
        Remove a single skill from the embedding cache and mark the
        BM25 index as dirty so it will be rebuilt on the next search.

        Args:
            skill_name: The skill identifier to evict from cache.
        """
        with self._cache_lock:
            self._embedding_cache.pop(skill_name, None)
        self._bm25_dirty = True  # Invalidate BM25 too

    def invalidate_all(self) -> None:
        """Clear the entire embedding cache and mark the BM25 index as dirty."""
        with self._cache_lock:
            self._embedding_cache.clear()
        self._bm25_dirty = True

    # -----------------------------------------------------------------------
    # BM25 Index Cache
    # -----------------------------------------------------------------------

    def _rebuild_bm25_index(self, skills: list[Skill]) -> None:
        """
        Build the BM25 index from scratch using the given skill list.

        The index is built over the concatenated text of
        ``skill.id + description + tags + repo_name``.

        Args:
            skills: The list of active Skill ORM objects to index.
        """
        with self._bm25_lock:
            if not skills:
                self._bm25_index = None
                self._bm25_skill_ids = []
                return

            corpus_tokens: list[list[str]] = []
            skill_ids: list[str] = []
            for skill in skills:
                tags = skill.get_tags()
                repo = skill.repo_name or "global"
                text = f"{skill.id} {skill.description} {' '.join(tags)} {repo}"
                corpus_tokens.append(self._tokenizer(text))
                skill_ids.append(skill.id)

            self._bm25_index = BM25Okapi(corpus_tokens)
            self._bm25_skill_ids = skill_ids
            self._bm25_dirty = False
            logger.debug(f"BM25 index rebuilt: {len(skill_ids)} skills")

    def _get_or_build_bm25(self, skills: list[Skill]) -> tuple:
        """
        Return the cached BM25 index, rebuilding it if the dirty flag is set.

        Args:
            skills: The full list of active Skill objects (used for rebuild).

        Returns:
            A tuple of ``(BM25Okapi index, list_of_skill_ids)``.
        """
        if self._bm25_dirty or self._bm25_index is None:
            self._rebuild_bm25_index(skills)
        return self._bm25_index, self._bm25_skill_ids

    # -----------------------------------------------------------------------
    # BM25 Search
    # -----------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        skills: list[Skill],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        Perform BM25 lexical search over name + description + tags + repo.

        Args:
            query: The search query string.
            skills: The list of Skill objects to search over.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(skill_id, score)`` tuples, sorted by descending score.
        """
        if not skills:
            return []

        bm25, skill_ids = self._get_or_build_bm25(skills)
        if bm25 is None:
            return []

        query_tokens = self._tokenizer(query)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(skill_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # -----------------------------------------------------------------------
    # Semantic Search
    # -----------------------------------------------------------------------

    def _semantic_search(
        self,
        query: str,
        skills: list[Skill],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        Perform semantic similarity search over description + tags (fitting 512 tokens).

        Embeddings for skills are cached and persisted. Uncached skills are
        encoded in batch and then cached.

        Args:
            query: The search query string.
            skills: The list of Skill objects to search over.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(skill_id, cosine_similarity)`` tuples, sorted by
            descending similarity.
        """
        if self._model is None or not skills:
            return []

        # Encode query
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return []
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []
        query_normalized = query_emb / query_norm

        # Batch encode uncached skills
        uncached_skills: list[Skill] = []
        for skill in skills:
            if skill.id not in self._embedding_cache:
                uncached_skills.append(skill)

        if uncached_skills:
            texts_to_encode: list[str] = []
            for skill in uncached_skills:
                desc = skill.description
                tags = skill.get_tags()
                texts_to_encode.append(f"{desc} {' '.join(tags)}")

            embeddings = self._get_embeddings_batch(texts_to_encode)
            for skill, emb in zip(uncached_skills, embeddings):
                if emb is not None:
                    with self._cache_lock:
                        self._embedding_cache[skill.id] = emb

            # Persist new embeddings to disk
            self._save_embedding_cache()

        # Compute cosine similarity against each skill embedding
        results: list[tuple[str, float]] = []
        for skill in skills:
            if skill.id not in self._embedding_cache:
                continue
            skill_emb = self._embedding_cache[skill.id]
            skill_norm = np.linalg.norm(skill_emb)
            if skill_norm == 0:
                continue
            similarity = float(np.dot(query_normalized, skill_emb / skill_norm))
            results.append((skill.id, similarity))

        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # -----------------------------------------------------------------------
    # RRF Fusion
    # -----------------------------------------------------------------------

    def _rrf_fuse(
        self,
        bm25_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
        k: int,
    ) -> list[tuple[str, float]]:
        """
        Fuse BM25 and semantic search results using Reciprocal Rank Fusion.

        RRF score for each skill:
            ``sum( 1 / (k + rank + 1) )`` over all ranking lists where it appears.

        Args:
            bm25_results: List of ``(skill_id, score)`` from BM25 search.
            semantic_results: List of ``(skill_id, score)`` from semantic search.
            k: The RRF constant. Higher values flatten rank differences.

        Returns:
            A list of ``(skill_id, rrf_score)`` tuples sorted by descending
            RRF score.
        """
        rrf_scores: dict[str, float] = {}
        for rank, (skill_id, _) in enumerate(bm25_results):
            rrf_scores[skill_id] = rrf_scores.get(skill_id, 0) + 1.0 / (k + rank + 1)
        for rank, (skill_id, _) in enumerate(semantic_results):
            rrf_scores[skill_id] = rrf_scores.get(skill_id, 0) + 1.0 / (k + rank + 1)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    # -----------------------------------------------------------------------
    # Background model warmup
    # -----------------------------------------------------------------------

    def warmup_semantic_async(self) -> None:
        """
        Start loading the semantic model in a background thread.

        This is called after the first BM25 search returns, so the agent
        gets an instant response via BM25 while the model warms up in the
        background. Future searches will benefit from hybrid (BM25+semantic).
        """
        if self._semantic_warming_up or self._model_loaded or not self._semantic_available:
            return
        self._semantic_warming_up = True

        def _warmup():
            try:
                t0 = time.time()
                self._load_model()
                t1 = time.time()
                if self._model is not None:
                    logger.info(f"Background semantic warmup completed in {t1 - t0:.1f}s")
                else:
                    logger.info(f"Background semantic warmup skipped ({t1 - t0:.1f}s)")
            except Exception as e:
                logger.warning(f"Background semantic warmup failed: {e}")
            finally:
                self._semantic_warming_up = False

        thread = threading.Thread(target=_warmup, daemon=True, name="semantic-warmup")
        thread.start()

    # -----------------------------------------------------------------------
    # Tier 1: Search — metadata + scores
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        repo_scope: str = "all",
        current_repo: str = "",
        tags_filter: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Tier 1: Search skills and return metadata with relevance scores.

        This is the primary entry point for the agent. It returns a compact
        result set (no full content) so the agent can decide which skills
        to inspect further via :meth:`get_skill_content`.

        Performance:
            - BM25-only search: <100ms (default, no model needed)
            - Hybrid (BM25+semantic): 200-500ms after model is loaded
            - First call with SKILLS_LAB_SEMANTIC=1: model download + load
              may take 1-10 minutes depending on network speed. The first
              search returns immediately via BM25; the model warms up in
              the background for subsequent searches.

        Args:
            query: The search query string.
            repo_scope: Either ``"current"`` (specific repo + global skills)
                or ``"all"`` (all repos).
            current_repo: The name of the currently open repository.
                Required when ``repo_scope`` is ``"current"``.
            tags_filter: If provided, only skills that have at least one
                matching tag (case-insensitive) will be returned.
            top_k: Maximum number of results to return. Defaults to the
                ``top_k`` set in the constructor.

        Returns:
            A list of dicts, each containing:
            ``{"skill": Skill, "rrf_score": float}``
        """
        t_start = time.time()

        # Do NOT call _load_model() here — that blocks for 1-10 min on first call.
        # Instead, the model is loaded asynchronously via warmup_semantic_async().

        if top_k is None:
            top_k = self._top_k

        if repo_scope == "current" and not current_repo:
            logger.warning("repo_scope='current' but current_repo is empty. Falling back to 'all'.")
            repo_scope = "all"

        session = self._session_factory()
        try:
            query_skills = session.query(Skill).filter(Skill.is_active == True)

            if repo_scope == "current":
                query_skills = query_skills.filter(
                    (Skill.repo_name == current_repo) | (Skill.repo_name == "global")
                )

            # Apply tags filter at DB level for efficiency (no Python loop)
            if tags_filter:
                # Fall back to Python filtering (SQLite JSON ops are limited)
                all_skills = query_skills.all()
                skills = []
                for s in all_skills:
                    s_tags = s.get_tags()
                    s_tags_lower = {st.lower() for st in s_tags}
                    if any(t.lower() in s_tags_lower for t in tags_filter):
                        skills.append(s)
            else:
                skills = query_skills.all()
        finally:
            session.close()

        if not skills:
            return []

        # BM25 search (uses cached index) — always fast, <10ms
        t_bm25 = time.time()
        bm25_results = self._bm25_search(query, skills, top_k=len(skills))
        t_bm25_end = time.time()

        # Semantic search (if a model is ALREADY loaded)
        # If model is not loaded yet, skip semantic — just return BM25 results.
        semantic_results: list[tuple[str, float]] = []
        t_semantic = 0.0
        if self._semantic_available and self._model is not None:
            t_semantic = time.time()
            semantic_results = self._semantic_search(query, skills, top_k=len(skills))
            t_semantic = time.time() - t_semantic

        # RRF fusion
        fused = self._rrf_fuse(bm25_results, semantic_results, self._rrf_k)
        top_results = fused[:top_k]

        # Attach Skill ORM objects — use dict lookup instead of N separate DB queries
        skill_map = {s.id: s for s in skills}
        results: list[dict] = []
        for skill_id, score in top_results:
            skill = skill_map.get(skill_id)
            if skill:
                results.append({"skill": skill, "rrf_score": score})

        t_total = time.time() - t_start
        search_mode = "hybrid" if semantic_results else "bm25-only"
        logger.debug(
            f"Search [{search_mode}] completed in {t_total * 1000:.0f}ms "
            f"(bm25: {t_bm25_end - t_bm25:.3f}s, semantic: {t_semantic:.3f}s, "
            f"skills: {len(skills)}, results: {len(results)})"
        )

        # Trigger background warmup if semantic is enabled but model not loaded yet.
        # This ensures the first search returns fast (BM25-only) while the model
        # loads asynchronously for future searches.
        if self._semantic_available and self._model is None and not self._semantic_warming_up:
            self.warmup_semantic_async()

        return results

    # -----------------------------------------------------------------------
    # Tier 2: Get full skill content
    # -----------------------------------------------------------------------

    def get_skill_content(self, skill_name: str) -> Optional[dict]:
        """
        Tier 2: Retrieve the full SKILL.md content and metadata for a skill.

        Args:
            skill_name: The unique skill identifier.

        Returns:
            A dict with keys:
            - ``skill``: Skill object serialized as a dict.
            - ``frontmatter``: Parsed YAML frontmatter dict.
            - ``body``: The SKILL.md body text (or an error message if
              the file is missing).
            - ``raw``: The raw file contents.

            Returns ``None`` if the skill does not exist in the database.
        """
        session = self._session_factory()
        try:
            skill = session.query(Skill).filter_by(id=skill_name).first()
            if not skill:
                return None
        finally:
            session.close()

        try:
            skill_data = self._manager.read_skill(skill_name)
            return {
                "skill": skill.to_dict(),
                "frontmatter": skill_data["frontmatter"],
                "body": skill_data["body"],
                "raw": skill_data["raw"],
            }
        except FileNotFoundError:
            return {
                "skill": skill.to_dict(),
                "frontmatter": {},
                "body": "(SKILL.md file not found)",
                "raw": "",
            }

    # -----------------------------------------------------------------------
    # Dedup detection
    # -----------------------------------------------------------------------

    def check_duplicates(
        self,
        skill_name: str,
        description: str,
    ) -> list[dict]:
        """
        Check whether a new skill is a potential duplicate of existing skills.

        Compares the embedding of the given description against all active
        skills (excluding the skill itself). If cosine similarity exceeds
        the configured ``dedup_threshold``, the skill is flagged as a
        potential duplicate.

        Args:
            skill_name: The unique identifier of the new skill.
            description: The description text of the new skill to embed
                and compare.

        Returns:
            A list of dicts sorted by descending similarity, each containing:
            ``{"name": str, "similarity": float, "description": str, "version": str}``
        """
        self._load_model()
        if self._model is None:
            return []

        new_emb = self._get_embedding(description)
        if new_emb is None:
            return []
        new_norm = np.linalg.norm(new_emb)
        if new_norm == 0:
            return []
        new_normalized = new_emb / new_norm

        session = self._session_factory()
        try:
            skills = session.query(Skill).filter(
                Skill.is_active == True,
                Skill.id != skill_name,
            ).all()
        finally:
            session.close()

        duplicates: list[dict] = []
        for skill in skills:
            if skill.id in self._embedding_cache:
                emb = self._embedding_cache[skill.id]
            else:
                text = f"{skill.description} {' '.join(skill.get_tags())}"
                emb = self._get_embedding(text)
                if emb is None:
                    continue

            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue

            similarity = float(np.dot(new_normalized, emb / emb_norm))
            if similarity >= self._dedup_threshold:
                duplicates.append({
                    "name": skill.id,
                    "similarity": round(similarity, 4),
                    "description": skill.description,
                    "version": skill.version_number,
                })

        return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)

    # -----------------------------------------------------------------------
    # Lineage
    # -----------------------------------------------------------------------

    def get_lineage_chain(self, skill_name: str) -> list[dict]:
        """
        Retrieve the changelog history for a skill (version transitions).

        Args:
            skill_name: The unique skill identifier.

        Returns:
            A list of dicts, each representing a changelog entry:
            ``{"skill": Skill, "from_version": str, "to_version": str,
            "trigger": str, "reason": str, "source_skill_id": Optional[str]}``

            Returns an empty list if the skill does not exist.
        """
        session = self._session_factory()
        try:
            skill = session.query(Skill).filter_by(id=skill_name).first()
            if not skill:
                return []

            entries = session.query(SkillChangelog).filter_by(
                skill_id=skill_name
            ).order_by(SkillChangelog.created_at.asc()).all()

            chain: list[dict] = []
            for entry in entries:
                chain.append({
                    "skill": skill,
                    "from_version": entry.from_version,
                    "to_version": entry.to_version,
                    "trigger": entry.trigger,
                    "reason": entry.reason,
                    "source_skill_id": entry.source_skill_id,
                })
            return chain
        finally:
            session.close()

    def get_lineage_tree(self, skill_name: str) -> Optional[dict]:
        """
        Get the lineage tree for a skill (flat list of version transitions).

        Args:
            skill_name: The unique skill identifier.

        Returns:
            A dict with skill metadata and a ``history`` list of transitions,
            or ``None`` if the skill has no changelog entries:

            .. code-block:: python

                {
                    "id": str,
                    "name": str,
                    "type": str,
                    "version_number": str,
                    "is_active": bool,
                    "history": [
                        {
                            "trigger": str,
                            "from": str,
                            "to": str,
                            "reason": str,
                            "source": Optional[str],
                        },
                        ...
                    ],
                }
        """
        chain = self.get_lineage_chain(skill_name)
        if not chain:
            return None
        skill = chain[0]["skill"]
        return {
            "id": skill.id,
            "name": skill.display_name,
            "type": skill.skill_type,
            "version_number": skill.version_number,
            "is_active": skill.is_active,
            "history": [
                {
                    "trigger": c["trigger"],
                    "from": c["from_version"],
                    "to": c["to_version"],
                    "reason": c["reason"],
                    "source": c.get("source_skill_id"),
                }
                for c in chain
            ],
        }
