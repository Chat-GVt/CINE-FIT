"""
Qwen3-Embedding-0.6B (ONNX, onnxruntime) 로 movie_db_scored.json 임베딩 생성.

BGE-M3(generate_embeddings.py)와 동일 포맷의 결과를 movie_embeddings_qwen3.json 으로 저장해
recommender.py / 평가 스크립트가 그대로 읽을 수 있게 한다.

Qwen3-Embedding 특징(BGE-M3와 다른 점):
- 디코더 기반이라 ONNX 입력에 position_ids + 빈 past_key_values(28층)가 필요하다.
  -> 입력 이름을 추측하지 않고 session.get_inputs()를 읽어 동적으로 채운다.
- 문장 임베딩은 last-token pooling(마지막 토큰 hidden state) 후 L2 정규화.
- instruction 접두어는 '쿼리'에만 붙인다. 영화 corpus는 '문서'라 접두어 없이 인코딩.
"""

import json
import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX"
ONNX_FILE = "onnx/model.onnx"          # fp32 (+ model.onnx_data 자동 동반 다운로드)
ONNX_DATA_FILE = "onnx/model.onnx_data"
TOKENIZER_FILE = "tokenizer.json"
MAX_LENGTH = 512
OUT_PATH = os.path.join(BASE_DIR, "movie_embeddings_qwen3.json")

_tokenizer = None
_session = None
_kv_specs = None   # [(name, num_heads, head_dim), ...] past_key_values 입력 스펙


def load_model():
    global _tokenizer, _session, _kv_specs
    if _session is not None:
        return _tokenizer, _session

    tok_path = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE)
    # model.onnx 는 외부 가중치(model.onnx_data)를 같은 폴더에서 참조하므로 둘 다 받아야 한다.
    hf_hub_download(repo_id=REPO_ID, filename=ONNX_DATA_FILE)
    model_path = hf_hub_download(repo_id=REPO_ID, filename=ONNX_FILE)

    _tokenizer = Tokenizer.from_file(tok_path)
    _tokenizer.enable_truncation(max_length=MAX_LENGTH)
    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # past_key_values.* 입력들의 (heads, head_dim) 고정 차원을 그래프에서 읽어둔다.
    # 형태 예: [batch, num_heads, past_seq_len, head_dim] -> past_seq_len=0 으로 빈 캐시 생성.
    reserved = {"input_ids", "attention_mask", "position_ids"}
    _kv_specs = []
    for inp in _session.get_inputs():
        if inp.name in reserved:
            continue
        shape = inp.shape  # 동적 축은 문자열/None
        num_heads = shape[1] if isinstance(shape[1], int) else 8
        head_dim = shape[3] if isinstance(shape[3], int) else 128
        _kv_specs.append((inp.name, num_heads, head_dim))
    return _tokenizer, _session


def embed_text(text: str) -> list:
    tokenizer, session = load_model()
    enc = tokenizer.encode(text)
    n = len(enc.ids)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    position_ids = np.arange(n, dtype=np.int64).reshape(1, n)

    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # 빈 past_key_values (past_seq_len=0). 모델이 요구하는 것만 채운다.
    input_names = {i.name for i in session.get_inputs()}
    for name, num_heads, head_dim in _kv_specs:
        if name in input_names:
            feeds[name] = np.zeros((1, num_heads, 0, head_dim), dtype=np.float32)

    outputs = session.run(["last_hidden_state"], feeds)
    last_hidden = outputs[0][0]          # (seq_len, hidden)
    vec = last_hidden[-1]                # last-token pooling (패딩 없음: 마지막이 실제 마지막 토큰)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def main():
    movies_path = os.path.join(BASE_DIR, "movie_db_scored.json")
    with open(movies_path, encoding="utf-8") as f:
        movies = json.load(f)

    print("Qwen3-Embedding-0.6B ONNX 모델 로딩 중... (최초 1회 ~2.4GB 다운로드)")
    load_model()
    print(f"입력 시그니처: {[i.name for i in _session.get_inputs()]}")

    embeddings = {}
    for i, m in enumerate(movies, 1):
        text = m["overview"] + " " + " ".join(m.get("keywords", []))
        vec = embed_text(text)
        embeddings[m["movie_id"]] = vec
        print(f"  [{i}/{len(movies)}] {m['korean_title']} -> {len(vec)}차원")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"\n{OUT_PATH} 저장 완료 ({len(embeddings)}편, dim={len(next(iter(embeddings.values())))})")


if __name__ == "__main__":
    main()
