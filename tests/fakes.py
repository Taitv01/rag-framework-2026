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


# Two short stories and a chat model scripted by prompt, for pipeline tests (see make_rag).
STORIES = {
    "thach_sanh.md": "# Thạch Sanh\n\nThạch Sanh sống dưới gốc đa, gia tài chỉ có một lưỡi búa của cha. "
                     "Chàng dùng búa chặt đầu chằn tinh, xác nó là con trăn khổng lồ.",
    "tam_cam.md": "# Tấm Cám\n\nBụt bảo Tấm thả cá bống xuống giếng và gọi bống lên ăn cơm. "
                  "Mẹ con Cám bắt bống làm thịt.",
}


class ScriptedChat:
    """Chat model whose reply depends on the prompt; records every call."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages, **_kwargs):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply(prompt))

    def stream(self, messages, **kwargs):
        for word in self.invoke(messages).content.split(" "):
            yield SimpleNamespace(content=word + " ")


def default_reply(prompt):
    if "search query optimizer" in prompt:
        return "Thạch Sanh giết chằn tinh bằng gì"
    if "document relevance grader" in prompt:
        return "S1"
    return "Thạch Sanh dùng búa [S1]."
