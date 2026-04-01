"""
Skills Lab — Model Manager

Handles downloading, caching, and checking availability of embedding models
used for semantic search.

Default model: BAAI/bge-small-en-v1.5 (~100MB)
Fallback model: sentence-transformers/all-MiniLM-L6-v2 (~80MB)

Backends (auto-selected in priority order):
  1. ONNX Runtime — CPU-only, no torch/GPU needed (~50MB extra)
     Uses ``onnxruntime`` + ``tokenizers`` directly. Zero torch dependency.
  2. PyTorch — requires torch + sentence-transformers (~2GB extra)

Cache locations checked (in order):
  1. Workspace local cache (workspace/.cache/models/)
  2. HuggingFace hub cache (~/.cache/huggingface/hub/)
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Backend priority: onnxruntime (CPU-only) > torch (CPU+GPU)
_BACKEND_PRIORITY = ["onnx", "torch"]


# ---------------------------------------------------------------------------
# Cache detection
# ---------------------------------------------------------------------------


def detect_backend() -> str:
    """
    Detect the best available embedding backend.

    Priority:
      1. onnxruntime — CPU-only, small footprint, no GPU/CUDA needed
      2. torch — heavier, but supports GPU acceleration

    Returns:
        ``"onnx"`` if onnxruntime (and tokenizers) are importable,
        ``"torch"`` if only torch is available,
        or an empty string if neither is available.
    """
    for backend in _BACKEND_PRIORITY:
        try:
            if importlib.util.find_spec(backend) is not None:
                if backend == "onnx":
                    # ONNX path also needs tokenizers and huggingface_hub
                    if importlib.util.find_spec("tokenizers") is None:
                        continue
                    if importlib.util.find_spec("onnx") is None:
                        continue
                logger.debug(f"Detected embedding backend: {backend}")
                return backend
        except Exception:
            continue
    logger.debug("No embedding backend detected (onnxruntime or torch)")
    return ""


def is_model_cached(model_name: str, workspace_path: str = "") -> bool:
    """
    Check whether an embedding model is already cached locally.

    Checks three locations (short-circuits on first match):
      1. Workspace local cache (fastest — local directory check)
      2. HuggingFace hub cache for ONNX models
      3. HuggingFace hub cache (generic scan)

    Performance: checks local workspace first (O(1)), then falls back
    to scanning HuggingFace cache directories only if needed.

    Args:
        model_name: The HuggingFace model identifier.
        workspace_path: Optional workspace root to check for local model cache.

    Returns:
        ``True`` if the model appears to be cached, ``False`` otherwise.
    """
    # Resolve to ONNX repo name for cache checking
    from core.onnx_encoder import ONNX_MODEL_MAP
    onnx_repo = ONNX_MODEL_MAP.get(model_name, model_name)
    safe_names = {
        model_name.replace("/", "--").lower(),
        onnx_repo.replace("/", "--").lower(),
    }

    # 0. Quick check: workspace local cache (fastest — no directory scanning)
    if workspace_path:
        for safe_name in safe_names:
            local_cache = os.path.join(workspace_path, ".cache", "models", safe_name)
            if os.path.isdir(local_cache):
                try:
                    if any(os.scandir(local_cache)):
                        return True
                except OSError:
                    pass

    # 1. HuggingFace hub cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf_cache):
        try:
            for entry in os.scandir(hf_cache):
                entry_lower = entry.name.lower()
                if any(sn in entry_lower for sn in safe_names):
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

    When the ONNX backend is detected, uses the pure ``OnnxEncoder``
    (onnxruntime + tokenizers) — zero torch dependency.
    When only the torch backend is available, falls back to
    ``sentence_transformers.SentenceTransformer``.

    Args:
        model_name: HuggingFace model identifier.
        workspace_path: Optional workspace path used for an alternative
            cache location check.

    Returns:
        ``True`` if the model was successfully loaded/downloaded,
        ``False`` otherwise.
    """
    logger.info(f"Checking embedding model: {model_name}")

    backend = detect_backend()
    if not backend:
        logger.warning(
            "No embedding backend found (onnxruntime or torch). Cannot load model."
        )
        return False

    if backend == "onnx":
        return _download_onnx(model_name, workspace_path)
    else:
        return _download_torch(model_name, workspace_path)


def _download_onnx(model_name: str, workspace_path: str = "") -> bool:
    """
    Download and verify an embedding model using the pure ONNX path.

    Uses ``OnnxEncoder`` which requires only ``onnxruntime`` + ``tokenizers``
    + ``huggingface_hub``.  No torch or sentence-transformers needed.
    """
    logger.info("Using ONNX backend (CPU-only, no GPU required)")

    if is_model_cached(model_name, workspace_path):
        logger.info(f"Model '{model_name}' is already cached (ONNX).")
    else:
        logger.info(f"Model '{model_name}' not found in cache — downloading ONNX model...")

    cache_dir = os.path.join(workspace_path, ".cache", "models") if workspace_path else ""

    try:
        from core.onnx_encoder import OnnxEncoder

        t0 = _time_monotonic()
        encoder = OnnxEncoder(model_name, cache_dir=cache_dir)
        elapsed = _time_monotonic() - t0

        # Sanity check: encode a test string
        test_vec = encoder.encode("test query", convert_to_numpy=True)
        if test_vec is not None and test_vec.size > 0:
            logger.info(
                f"Model '{model_name}' ready (ONNX) in {elapsed:.1f}s "
                f"(embedding dim={test_vec.shape[0]})"
            )
            return True
        else:
            logger.warning(f"Model '{model_name}' loaded but produced empty embedding.")
            return False
    except Exception as e:
        logger.error(f"Failed to download/load ONNX model '{model_name}': {e}")
        return False


def _download_torch(model_name: str, workspace_path: str = "") -> bool:
    """
    Download and verify an embedding model using the torch backend.

    Requires ``sentence_transformers`` and ``torch`` to be installed.
    """
    logger.info("Using PyTorch backend")

    # Fast probe — is sentence_transformers importable?
    try:
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.warning(
                "sentence-transformers is not installed. "
                "Cannot download embedding model. "
                "Install with: pip install 'skills-lab[semantic]'"
            )
            return False
    except Exception:
        pass

    # Disable CUDA if not available to avoid crashes
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("PyTorch CPU backend (GPU not available)")
    except Exception:
        logger.info("PyTorch backend")

    if is_model_cached(model_name, workspace_path):
        logger.info(f"Model '{model_name}' is already cached.")
    else:
        logger.info(f"Model '{model_name}' not found in cache — downloading...")

    try:
        from sentence_transformers import SentenceTransformer

        t0 = _time_monotonic()
        model = SentenceTransformer(model_name)
        elapsed = _time_monotonic() - t0
        logger.info(f"Model '{model_name}' loaded in {elapsed:.1f}s (torch)")

        test_vec = model.encode("test query", convert_to_numpy=True)
        if test_vec is not None and test_vec.size > 0:
            logger.info(
                f"Model '{model_name}' ready "
                f"(embedding dim={test_vec.shape[0]}, backend=torch)"
            )
            return True
        else:
            logger.warning(f"Model '{model_name}' loaded but produced empty embedding.")
            return False
    except Exception as e:
        logger.error(f"Failed to download/load model '{model_name}' (torch): {e}")
        return False


def _time_monotonic() -> float:
    """"Monotonic timer for measuring model load time."""
    import time
    return time.monotonic()


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
