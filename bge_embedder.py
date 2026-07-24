"""
BGE-M3 임베딩 공용 모듈 (onnxruntime 기반, torch 불필요).
generate_embeddings.py(B2, 배치)와 recommender.py(B6, 요청 시점 자연어 임베딩)에서 공유.
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MAX_LENGTH = 512

_tokenizer = None
_session = None


def load_model():
    """모델을 최초 1회만 로드하고 이후엔 캐시된 걸 재사용."""
    global _tokenizer, _session
    if _tokenizer is None or _session is None:
        tok_path = hf_hub_download(repo_id="BAAI/bge-m3", filename="onnx/tokenizer.json")
        model_path = hf_hub_download(repo_id="BAAI/bge-m3", filename="onnx/model.onnx")
        hf_hub_download(repo_id="BAAI/bge-m3", filename="onnx/model.onnx_data")

        _tokenizer = Tokenizer.from_file(tok_path)
        _tokenizer.enable_truncation(max_length=MAX_LENGTH)
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _tokenizer, _session


def embed_text(text: str) -> list:
    tokenizer, session = load_model()
    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    _, sentence_embedding = session.run(
        ["token_embeddings", "sentence_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    vec = sentence_embedding[0]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()
