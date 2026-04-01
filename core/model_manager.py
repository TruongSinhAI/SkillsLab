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
     WARNING: May crash on Windows machines without GPU/C++ redistributable.

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
_BACKEND_PRIORITY = ["onnxruntime", "torch"]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def detect_backend() -> str:
    """
    Detect the best available embedding backend.

    Priority:
      1. onnxruntime — CPU-only, small footprint, no GPU/CUDA needed
      2. torch — heavier, but supports GPU acceleration

    Unlike ``importlib.util.find_spec()``, this function actually tries to
    *import* the package to catch DLL load failures (common on Windows
    without VC++ Redistributable).

    Returns:
        ``"onnxruntime"`` if onnxruntime (and tokenizers) are importable,
        ``"torch"`` if only torch is available,
        or an empty string if neither is available.
    """
    for backend in _BACKEND_PRIORITY:
        try:
            # Actually try to import — find_spec() only checks metadata,
            # not whether the native DLLs can load (catches c10.dll / pybind11 errors)
            __import__(backend)
            if backend == "onnxruntime":
                # ONNX path also needs tokenizers and huggingface_hub
                if importlib.util.find_spec("tokenizers") is None:
                    logger.debug("ONNX backend: tokenizers not installed")
                    continue
                if importlib.util.find_spec("huggingface_hub") is None:
                    logger.debug("ONNX backend: huggingface_hub not installed")
                    continue
            logger.debug(f"Detected embedding backend: {backend}")
            return backend
        except ImportError:
            continue
        except OSError:
            # DLL load failed (common on Windows) — skip this backend
            logger.debug(f"Backend '{backend}' found but DLL load failed")
            continue
        except Exception:
            continue
    logger.debug("No embedding backend detected (onnxruntime or torch)")
    return ""


def check_onnx_deps() -> dict[str, bool]:
    """
    Check which ONNX dependencies are actually importable.

    Unlike ``find_spec()``, this tests real imports so DLL errors
    are properly detected.

    Returns:
        A dict with keys ``onnxruntime``, ``tokenizers``, ``huggingface_hub``
        and boolean values indicating actual importability.
    """
    deps = {}
    for pkg in ("onnxruntime", "tokenizers", "huggingface_hub"):
        try:
            __import__(pkg)
            deps[pkg] = True
        except Exception:
            deps[pkg] = False
    return deps


# ---------------------------------------------------------------------------
# Cache detection
# ---------------------------------------------------------------------------


def is_model_cached(model_name: str, workspace_path: str = "") -> bool:
    """
    Check whether an embedding model is already cached locally.

    Checks multiple locations (short-circuits on first match):
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

    Strategy:
      1. If ONNX deps are available → use pure OnnxEncoder (zero torch)
      2. If only torch is available → warn about potential crash, try it
      3. If torch also fails (e.g. c10.dll crash) → return False with clear message

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
    if backend == "onnxruntime":
        return _download_onnx(model_name, workspace_path)
    elif backend == "torch":
        return _download_torch(model_name, workspace_path)
    else:
        # No backend at all
        deps = check_onnx_deps()
        missing = [k for k, v in deps.items() if not v]
        logger.warning(
            f"No embedding backend found. Missing packages: {', '.join(missing)}. "
            f"Install with: pip install onnxruntime tokenizers huggingface_hub"
        )
        return False


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
    except ImportError as e:
        error_str = str(e)
        if "onnxruntime" in error_str and ("DLL" in error_str or "pybind11" in error_str):
            logger.error(
                f"onnxruntime DLL load failed: {e}\n"
                f"This means Microsoft Visual C++ Redistributable is missing or corrupted.\n"
                f"FIX (choose one):\n"
                f"  1. Download and install VC++ Redistributable:\n"
                f"     https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"  2. Then reinstall onnxruntime:\n"
                f"     pip uninstall onnxruntime -y && pip install onnxruntime\n"
                f"  3. Then retry: skills-lab download-model"
            )
        else:
            logger.error(f"ONNX backend missing dependency: {e}")
        return False
    except OSError as e:
        error_str = str(e)
        if "DLL" in error_str or "pybind11" in error_str:
            logger.error(
                f"onnxruntime DLL load failed: {e}\n"
                f"This means Microsoft Visual C++ Redistributable is missing or corrupted.\n"
                f"FIX (choose one):\n"
                f"  1. Download and install VC++ Redistributable:\n"
                f"     https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"  2. Then reinstall onnxruntime:\n"
                f"     pip uninstall onnxruntime -y && pip install onnxruntime\n"
                f"  3. Then retry: skills-lab download-model"
            )
        else:
            logger.error(f"Failed to download/load ONNX model '{model_name}': {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to download/load ONNX model '{model_name}': {e}")
        return False


def _download_torch(model_name: str, workspace_path: str = "") -> bool:
    """
    Download and verify an embedding model using the torch backend.

    Requires ``sentence_transformers`` and ``torch`` to be installed.

    WARNING: This may crash on Windows machines without GPU or C++
    redistributable (c10.dll error).  ONNX backend is strongly recommended.
    """
    logger.warning(
        "Using PyTorch backend — this may crash on machines without GPU! "
        "For best compatibility, install ONNX deps instead: "
        "pip install onnxruntime tokenizers huggingface_hub"
    )

    # Fast probe — is sentence_transformers importable?
    try:
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.warning(
                "sentence-transformers is not installed. "
                "Cannot use torch backend. "
                "Install ONNX deps instead: pip install onnxruntime tokenizers huggingface_hub"
            )
            return False
    except Exception:
        pass

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
    except OSError as e:
        # Catch Windows DLL errors (c10.dll) and give actionable message
        error_str = str(e)
        if "c10.dll" in error_str or "DLL" in error_str or "1114" in error_str:
            logger.error(
                f"PyTorch DLL error (common on Windows without GPU): {e}\n"
                f"SOLUTION: Uninstall torch and use ONNX backend instead:\n"
                f"  pip uninstall torch sentence-transformers\n"
                f"  pip install onnxruntime tokenizers huggingface_hub\n"
                f"Then run: skills-lab download-model"
            )
        else:
            logger.error(f"Failed to download/load model '{model_name}' (torch): {e}")
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
