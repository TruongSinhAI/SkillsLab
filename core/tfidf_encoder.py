"""
Skills Lab — Pure NumPy TF-IDF Encoder

Zero native dependencies — works on ANY machine (including Windows
without VC++ Redistributable). Only requires ``numpy`` (already a core dep).

Uses TF-IDF (Term Frequency–Inverse Document Frequency) to create
sparse vector embeddings, then computes cosine similarity. Not as
powerful as transformer-based models, but works reliably everywhere
and provides meaningful semantic matching for skill search.

This is the last-resort fallback when both onnxruntime and torch
are unavailable or broken (e.g. DLL load failures).
"""

import logging
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class TfidfEncoder:
    """
    Pure numpy TF-IDF sentence encoder. Compatible interface with
    ``SentenceTransformer.encode()`` and ``OnnxEncoder.encode()``.

    Parameters
    ----------
    model_name_or_path:
        Ignored (kept for API compatibility).
    cache_dir:
        Ignored (kept for API compatibility).
    """

    def __init__(self, model_name_or_path: str = "", cache_dir: str = ""):
        self.max_seq_length = 512
        self._needs_pooling = False  # Already outputs 2D

        # IDF cache: word → idf_score
        self._idf: dict[str, float] = {}
        self._idf_vector: np.ndarray | None = None
        self._vocab: dict[str, int] = {}  # word → index
        self._vocab_locked = False
        self._dim = 384  # Output dimension (padded/truncated for compatibility)

        # Train IDF on a small set of skill-related seed text
        self._train_seed_idf()

        logger.info(
            f"TfidfEncoder ready (pure numpy, {self._dim}d, "
            f"vocab size={len(self._vocab)})"
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
        Encode texts into normalized TF-IDF vectors.

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
            Ignored (kept for API compatibility).

        Returns
        -------
        np.ndarray of shape ``(n, dim)`` for list input, or ``(dim,)`` for string.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Update IDF with new words if vocab not locked
        if not self._vocab_locked:
            self._update_idf(texts)

        # Build TF-IDF vectors
        vectors = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            max_tf = max(tf.values())
            for word, count in tf.items():
                if word in self._vocab:
                    idx = self._vocab[word]
                    # Sublinear TF: 1 + log(tf)
                    tf_score = 1.0 + math.log(count) if count > 0 else 0
                    if max_tf > 0:
                        tf_score /= (1.0 + math.log(max_tf))
                    vectors[i, idx] = tf_score * self._idf.get(word, 1.0)

        # Normalize
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            vectors = vectors / norms

        # Pad or truncate to self._dim for API compatibility
        vectors = self._pad_or_truncate(vectors)

        if single_input:
            return vectors[0]
        return vectors

    def _pad_or_truncate(self, vectors: np.ndarray) -> np.ndarray:
        """Ensure output has consistent dimension."""
        current_dim = vectors.shape[1]
        if current_dim == self._dim:
            return vectors
        elif current_dim < self._dim:
            # Pad with zeros (cosine sim won't be affected much)
            padded = np.zeros((vectors.shape[0], self._dim), dtype=np.float32)
            padded[:, :current_dim] = vectors
            return padded
        else:
            # Truncate
            return vectors[:, :self._dim]

    # ------------------------------------------------------------------
    # TF-IDF internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        # Split on non-alphanumeric
        tokens = re.findall(r'[a-z0-9]+', text)
        return tokens

    def _train_seed_idf(self):
        """Train IDF on a small set of skill-related seed documents."""
        seeds = [
            "fix cors error api routes nextjs credentials fetch request",
            "docker multi-stage build optimize nodejs production image",
            "git conventional commits changelog semantic versioning",
            "react hooks usestate useeffect component lifecycle management",
            "python sqlalchemy database orm migration query optimization",
            "typescript interface generic type utility function declaration",
            "kubernetes deployment helm chart service ingress configuration",
            "css flexbox grid layout responsive design media query breakpoint",
            "authentication jwt token oauth2 login session management",
            "testing unit integration jest pytest mock stub coverage",
            "api rest graphql endpoint middleware error handling validation",
            "git branch merge rebase conflict resolution workflow strategy",
            "performance optimization lazy loading caching memoization bundle",
            "security xss csrf injection validation sanitization encryption",
            "debugging breakpoint logging error trace profiling monitoring",
        ]
        all_tokens = []
        doc_freq = Counter()
        for seed in seeds:
            tokens = self._tokenize(seed)
            all_tokens.extend(tokens)
            for word in set(tokens):
                doc_freq[word] += 1

        # Build vocab
        word_set = set(all_tokens)
        self._vocab = {word: idx for idx, word in enumerate(sorted(word_set))}
        n_docs = len(seeds)

        # IDF: log((1 + n) / (1 + df)) + 1  (smooth IDF)
        self._idf = {}
        for word, df in doc_freq.items():
            self._idf[word] = math.log((1 + n_docs) / (1 + df)) + 1

    def _update_idf(self, texts: list[str]):
        """
        Update IDF scores with new documents.
        Called during first batch to expand vocabulary.
        After first call, vocab is locked for consistency.
        """
        if self._vocab_locked:
            return

        new_words = set()
        for text in texts:
            tokens = self._tokenize(text)
            for word in tokens:
                if word not in self._vocab:
                    new_words.add(word)

        if not new_words:
            self._vocab_locked = True
            return

        # Add new words to vocab
        for word in sorted(new_words):
            self._vocab[word] = len(self._vocab)
            self._idf[word] = 1.0  # Default IDF for unseen words

        self._vocab_locked = True
        logger.debug(
            f"TfidfEncoder vocab expanded: {len(self._vocab)} words (locked)"
        )
