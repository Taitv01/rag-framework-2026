"""Model-free stand-ins shared by tests."""

import math
import re
import zlib
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Bag-of-words vectors hashed into a few dimensions: texts sharing words are close."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.documents_embedded = 0

    def _vector(self, text: str):
        vector = [0.0] * self.dim
        for token in re.findall(r"\w+", text.casefold()):
            vector[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def fake_embeddings_manager(dim: int = 256):
    """Object shaped like EmbeddingsManager for VectorStoreManager."""
    embeddings = HashEmbeddings(dim)
    return SimpleNamespace(
        embeddings=embeddings,
        embed_query=embeddings.embed_query,
        embed_documents=embeddings.embed_documents,
    )
