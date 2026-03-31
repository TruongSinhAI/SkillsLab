"""
Skills Lab — Model Manager

Handles downloading, caching, and checking availability of embedding models
used for semantic search.

Default model: BAAI/bge-small-en-v1.5 (~100MB)
Fallback model: sentence-transformers/all-MiniLM-L6-v2 (~80MB)

Cache locations checked (in order):
  1. sentence-transformers default cache (typically ~/.cache/torch/sentence_transformers/)
  2. HuggingFace hub cache (~/.cache/huggingface/hub/)
  3. Workspace local cache (workspace/.cache/models/)
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Cache detection
# ---------------------------------------------------------------------------


def is_model_cached(model_name: str, workspace_path: str = "") -> bool:
    """
    Check whether an embedding model is already cached locally.

    Checks three locations (short-circuits on first match):
      1. Workspace local cache (fastest — local directory check)
      2. Sentence-transformers cache
      3. HuggingFace hub cache

    Performance: checks local workspace first (O(1)), then falls back
    to scanning HuggingFace cache directories only if needed.

    Args:
        model_name: The HuggingFace model identifier (e.g. ``BAAI/bge-small-en-v1.5``).
        workspace_path: Optional workspace root to check for local model cache.

    Returns:
        ``True`` if the model appears to be cached, ``False`` otherwise.
    """
    # Normalise the model name for filesystem matching
    safe_name = model_name.replace("/", "--").lower()

    # 0. Quick check: workspace local cache (fastest — no directory scanning)
    if workspace_path:
        local_cache = os.path.join(workspace_path, ".cache", "models", safe_name)
        if os.path.isdir(local_cache):
            try:
                if any(os.scandir(local_cache)):
                    return True
            except OSError:
                pass

    # 1. Sentence-transformers cache (smaller, check first)
    st_cache = os.path.expanduser("~/.cache/torch/sentence_transformers")
    if os.path.isdir(st_cache):
        try:
            for entry in os.scandir(st_cache):
                if safe_name in entry.name.lower() or model_name.lower() in entry.name.lower():
                    return True
        except OSError:
            pass

    # 2. HuggingFace hub cache (can be very large — scan last)
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf_cache):
        try:
            for entry in os.scandir(hf_cache):
                if safe_name in entry.name.lower():
                    snapshot_dir = os.path.join(hf_cache, entry.name, "snapshots")
                    if os.path.isdir(snapshot_dir):
                        try:
                            if any(os.scandir(snapshot_dir)):
                                return True
                        except OSError:
                            pass
        except OSError:
            pass

    return False


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def download_embedding_model(
    model_name: str = DEFAULT_MODEL,
    workspace_path: str = "",
) -> bool:
    """
    Download and cache an embedding model for semantic search.

    Uses ``sentence_transformers.SentenceTransformer`` to download (or load
    from cache) the model.  If *sentence-transformers* is not installed the
    function logs a warning and returns ``False`` gracefully.

    Args:
        model_name: HuggingFace model identifier.
        workspace_path: Optional workspace path used for an alternative
            cache location check.

    Returns:
        ``True`` if the model was successfully loaded/downloaded,
        ``False`` otherwise.
    """
    logger.info(f"Checking embedding model: {model_name}")

    # Fast probe — is sentence_transformers importable?
    try:
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.warning(
                "sentence-transformers is not installed. "
                "Cannot download embedding model. "
                "Install with: pip install sentence-transformers torch"
            )
            return False
    except Exception:
        pass

    # Check if already cached
    if is_model_cached(model_name, workspace_path):
        logger.info(f"Model '{model_name}' is already cached.")
        # Still load it once to verify the cache is valid
    else:
        logger.info(f"Model '{model_name}' not found in cache — downloading...")

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model '{model_name}' (downloading if needed)...")
        model = SentenceTransformer(model_name)
        # Quick sanity check: encode a test string
        test_vec = model.encode("test query", convert_to_numpy=True)
        if test_vec is not None and test_vec.size > 0:
            logger.info(
                f"Model '{model_name}' ready "
                f"(embedding dim={test_vec.shape[0]})"
            )
            return True
        else:
            logger.warning(f"Model '{model_name}' loaded but produced empty embedding.")
            return False
    except Exception as e:
        logger.error(f"Failed to download/load model '{model_name}': {e}")
        return False


def download_with_fallback(workspace_path: str = "") -> tuple[bool, str]:
    """
    Try downloading the default model; fall back to the backup model.

    Args:
        workspace_path: Optional workspace path.

    Returns:
        A tuple ``(success, model_name)`` where *success* is ``True`` if
        a model was loaded and *model_name* is the name of the model
        that succeeded (or the last one that was tried).
    """
    for model in [DEFAULT_MODEL, FALLBACK_MODEL]:
        if download_embedding_model(model, workspace_path):
            return True, model
    return False, FALLBACK_MODEL
