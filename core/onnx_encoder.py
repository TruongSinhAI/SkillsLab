"""
Skills Lab — Pure ONNX Embedding Encoder

Drop-in replacement for ``sentence_transformers.SentenceTransformer`` that uses
ONLY ``onnxruntime`` + ``tokenizers`` + ``numpy``.  Zero dependency on ``torch``,
``transformers``, or ``sentence_transformers`` — works on any machine (including
Windows without GPU / C++ redistributable).

Architecture:
  1. Tokenize text with ``tokenizers.Tokenizer`` (Rust-based, standalone)
  2. Run ONNX inference with ``onnxruntime.InferenceSession``
  3. Mean-pool + L2-normalise the token embeddings with ``numpy``

Model files are downloaded from HuggingFace Hub repos that host pre-exported
ONNX weights (e.g. Xenova/* repos).  This eliminates the need for a PyTorch→ONNX
conversion step at runtime.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model-name mapping: sentence-transformers name → ONNX Hub repo
#
# Xenova repos contain tokenizer.json + model.onnx (pre-exported, no torch).
# Users can also pass an ONNX repo name directly.
# ---------------------------------------------------------------------------
ONNX_MODEL_MAP: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Xenova/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2": "Xenova/all-MiniLM-L6-v2",
}

# Known model properties (dim, max_seq_length)
_MODEL_PROPS: dict[str, dict] = {
    "Xenova/bge-small-en-v1.5": {"dim": 384, "max_seq_length": 512},
    "Xenova/all-MiniLM-L6-v2": {"dim": 384, "max_seq_length": 256},
    "BAAI/bge-small-en-v1.5": {"dim": 384, "max_seq_length": 512},
    "sentence-transformers/all-MiniLM-L6-v2": {"dim": 384, "max_seq_length": 256},
}


class OnnxEncoder:
    """
    Pure-ONNX sentence encoder.  Compatible interface with
    ``SentenceTransformer.encode()``.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace repo id containing ``model.onnx`` (or ``onnx/model.onnx``)
        and ``tokenizer.json``, or a local directory path.
    cache_dir:
        Optional HuggingFace Hub cache directory.
    """

    def __init__(self, model_name_or_path: str, cache_dir: str = ""):
        self._model_name = model_name_or_path
        self._cache_dir = cache_dir

        # Resolve to ONNX repo name if a sentence-transformers name is given
        onnx_repo = ONNX_MODEL_MAP.get(model_name_or_path, model_name_or_path)

        props = _MODEL_PROPS.get(onnx_repo, {})
        self.max_seq_length: int = props.get("max_seq_length", 512)

        # --- Download / locate ONNX model file ---
        onnx_path = self._resolve_onnx_path(onnx_repo, cache_dir)
        logger.info(f"Loading ONNX model from: {onnx_path}")

        import onnxruntime as ort
        # Use CPU provider explicitly — no GPU needed
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        logger.debug(f"ONNX inputs: {self._input_names}")
        logger.debug(f"ONNX outputs: {self._output_names}")

        # --- Load tokenizer ---
        self._tokenizer = self._resolve_tokenizer(onnx_repo, cache_dir)
        # Determine padding token from tokenizer
        pad_token = self._get_pad_token()
        self._tokenizer.enable_truncation(max_length=self.max_seq_length)
        self._tokenizer.enable_padding(
            pad_id=self._pad_id,
            pad_token=pad_token,
            length=self.max_seq_length,
        )

        # Auto-detect embedding dimension from ONNX model
        out_shape = self._session.get_outputs()[0].shape
        if len(out_shape) == 3:
            # (batch, seq_len, hidden_dim) → need pooling
            self._needs_pooling = True
        elif len(out_shape) == 2:
            # (batch, hidden_dim) → already pooled
            self._needs_pooling = False
        else:
            self._needs_pooling = True  # safe default

        logger.info(
            f"OnnxEncoder ready: model={onnx_repo}, "
            f"max_seq_length={self.max_seq_length}, needs_pooling={self._needs_pooling}"
        )

    # ------------------------------------------------------------------
    # Public API (SentenceTransformer-compatible)
    # ------------------------------------------------------------------

    def encode(
        self,
        texts,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode texts into normalized embedding vectors.

        Parameters
        ----------
        texts: str or list[str]
            One or more texts to encode.
        convert_to_numpy: bool
            Always returns numpy array (kept for API compatibility).
        show_progress_bar: bool
            Ignored (kept for API compatibility).
        normalize_embeddings: bool
            L2-normalise the output embeddings.
        batch_size: int
            Max texts per ONNX batch.

        Returns
        -------
        np.ndarray
            Shape ``(n, dim)`` for list input, or ``(dim,)`` for string input.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self._encode_batch(batch, normalize_embeddings)
            all_embeddings.append(emb)

        result = np.vstack(all_embeddings)

        if single_input:
            return result[0]
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_batch(self, texts: list[str], normalize: bool) -> np.ndarray:
        """Tokenize and run ONNX inference for a single batch."""
        encoded = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

        # Build ONNX inputs (only include what the model expects)
        inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = token_type_ids

        outputs = self._session.run(self._output_names, inputs)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden) or (batch, hidden)

        if self._needs_pooling:
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_emb = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            token_embeddings = sum_emb / sum_mask

        if normalize:
            norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            token_embeddings = token_embeddings / norms

        return token_embeddings

    # ------------------------------------------------------------------
    # Model / tokenizer file resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_onnx_path(repo: str, cache_dir: str = "") -> str:
        """
        Download (or locate cached) ONNX model file.

        Tries multiple paths:
          1. Local directory: ``{repo}/model.onnx``
          2. Local directory: ``{repo}/onnx/model.onnx``
          3. HF Hub: ``{repo}/model.onnx``
          4. HF Hub: ``{repo}/onnx/model.onnx``
        """
        # If it's a local directory, check directly
        if os.path.isdir(repo):
            for candidate in ("model.onnx", "onnx/model.onnx"):
                path = os.path.join(repo, candidate)
                if os.path.isfile(path):
                    return path
            raise FileNotFoundError(
                f"No ONNX model found in '{repo}'. "
                "Expected model.onnx or onnx/model.onnx"
            )

        # Download from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            for filename in ("model.onnx", "onnx/model.onnx"):
                try:
                    path = hf_hub_download(
                        repo,
                        filename=filename,
                        cache_dir=cache_dir or None,
                    )
                    if path and os.path.isfile(path):
                        logger.info(f"Downloaded ONNX model: {path}")
                        return path
                except Exception:
                    continue
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download ONNX models. "
                "Install with: pip install huggingface_hub"
            )

        raise FileNotFoundError(
            f"Could not find or download ONNX model from '{repo}'. "
            f"Tried: model.onnx, onnx/model.onnx"
        )

    @staticmethod
    def _resolve_tokenizer(repo: str, cache_dir: str = ""):
        """
        Load a fast tokenizer from a local directory or download from HF Hub.
        """
        from tokenizers import Tokenizer

        if os.path.isdir(repo):
            path = os.path.join(repo, "tokenizer.json")
            if os.path.isfile(path):
                return Tokenizer.from_file(path)
            raise FileNotFoundError(
                f"No tokenizer.json found in '{repo}'"
            )

        # from_pretrained downloads from HuggingFace Hub
        return Tokenizer.from_pretrained(repo, cache_dir=cache_dir or None)

    def _get_pad_token(self) -> str:
        """Detect the padding token from the tokenizer."""
        try:
            # Try to decode pad_id
            if hasattr(self._tokenizer, "model") and hasattr(self._tokenizer.model, "pad"):
                pad_id = self._tokenizer.model.pad
                if pad_id is not None and pad_id != 0:
                    # Try to get the token string
                    if hasattr(self._tokenizer.id_to_token, "__call__"):
                        token_str = self._tokenizer.id_to_token(pad_id)
                        if token_str:
                            self._pad_id = pad_id
                            return token_str
        except Exception:
            pass

        # Default: use [PAD] with id=0
        self._pad_id = 0
        return "[PAD]"
