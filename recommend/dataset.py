"""
데이터 계층: JSON 로딩 + BGE-M3 임베딩 + 영화 임베딩 캐시.

  [1] JSON 입출력
  [2] BGE-M3 임베딩 (onnxruntime 기반, torch 불필요)
  [3] 영화 임베딩 캐시 - get_movie_embeddings() 하나로 로드/생성 자동 처리
"""

import hashlib
import json
import os
from typing import Dict, List

from config import (
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_REPO,
    MOVIE_DB_PATH,
    MOVIE_EMBEDDING_PATH,
    QUESTION_PATH,
    TYPE_PATH,
    USER_INPUT_PATH,
)

# ============================================================
# [1] JSON 입출력
# ============================================================


def read_json(path: str, default=None):
    """파일이 없으면 default 반환 (default=None이면 예외)."""
    if not os.path.exists(path):
        if default is None:
            raise FileNotFoundError(f"파일이 없습니다: {path}")
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_movies() -> List[Dict]:
    """영화 DB (MVTI 4축 점수까지 계산된 상태)."""
    return read_json(MOVIE_DB_PATH)


def load_questions() -> List[Dict]:
    """MVTI 설문 문항."""
    return read_json(QUESTION_PATH)["questions"]


def load_types() -> Dict[str, Dict]:
    """16유형 정의 {유형코드: {"name": ..., "characters": [...]}}."""
    return read_json(TYPE_PATH)


def load_user_input() -> Dict:
    """
    사용자 입력 (설문 응답 + 자연어 취향).
    실제 서비스에서는 프론트에서 넘어오는 payload가 이 형태면 된다.
      {"answers": {"Q01": 4, ...}, "natural_language": "..."}
    """
    return read_json(USER_INPUT_PATH)


# ============================================================
# [2] BGE-M3 임베딩
# ============================================================

_tokenizer = None
_session = None


def load_model():
    """모델을 최초 1회만 로드하고 이후엔 캐시 재사용. import도 이때 처리."""
    global _tokenizer, _session
    if _tokenizer is None or _session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok_path = hf_hub_download(repo_id=EMBEDDING_REPO, filename="onnx/tokenizer.json")
        model_path = hf_hub_download(repo_id=EMBEDDING_REPO, filename="onnx/model.onnx")
        hf_hub_download(repo_id=EMBEDDING_REPO, filename="onnx/model.onnx_data")

        _tokenizer = Tokenizer.from_file(tok_path)
        _tokenizer.enable_truncation(max_length=EMBEDDING_MAX_LENGTH)
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _tokenizer, _session


def embed_text(text: str) -> List[float]:
    """문장 하나를 1024차원 L2 정규화 벡터로. 영화/사용자 입력 모두 이 함수를 쓴다."""
    import numpy as np

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


# ============================================================
# [3] 영화 임베딩 캐시
# ============================================================
# 캐시 파일 형식 (data/movie_embeddings.json):
#   {"model": "BAAI/bge-m3", "items": {"1": {"hash": "...", "vector": [...]}}}
# 텍스트 해시를 같이 저장해서 줄거리/키워드가 갱신되면 그 영화만 자동 재임베딩한다.


def build_movie_text(movie: Dict) -> str:
    """임베딩에 넣을 텍스트 = 줄거리 + 키워드."""
    return movie["overview"] + " " + " ".join(movie.get("keywords", []))


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> Dict[int, Dict]:
    raw = read_json(MOVIE_EMBEDDING_PATH, default={})
    if not raw:
        return {}

    if "items" in raw:  # 신규 형식
        if raw.get("model") != EMBEDDING_REPO:
            print(f"[embedding] 캐시 모델({raw.get('model')})이 달라 전체 재생성합니다.")
            return {}
        return {int(k): v for k, v in raw["items"].items()}

    # 구버전 형식({movie_id: vector})도 그대로 재사용 (해시가 없으니 유효한 걸로 간주)
    return {int(k): {"hash": None, "vector": v} for k, v in raw.items()}


def _is_valid(entry: Dict, text_hash: str) -> bool:
    if not entry or not entry.get("vector"):
        return False
    return entry.get("hash") in (None, text_hash)


def get_movie_embeddings(
    movies: List[Dict],
    force_rebuild: bool = False,
    verbose: bool = True,
) -> Dict[int, List[float]]:
    """
    {movie_id: 1024차원 벡터} 반환.
    캐시가 있으면 로드, 없거나 텍스트가 바뀐 영화만 새로 임베딩해서 저장한다.
    """
    cache = {} if force_rebuild else _load_cache()

    texts = {m["movie_id"]: build_movie_text(m) for m in movies}
    hashes = {mid: _text_hash(t) for mid, t in texts.items()}
    stale = [m for m in movies if not _is_valid(cache.get(m["movie_id"]), hashes[m["movie_id"]])]

    if not stale:
        if verbose:
            print(f"[embedding] 캐시 재사용 ({len(movies)}편, 재임베딩 없음)")
        return {mid: cache[mid]["vector"] for mid in texts if mid in cache}

    if verbose:
        print(f"[embedding] 새로 임베딩할 영화 {len(stale)}편 / 전체 {len(movies)}편")
        print("[embedding] BGE-M3 모델 로딩 중... (최초 1회는 다운로드로 몇 분 걸립니다)")
    load_model()

    for i, movie in enumerate(stale, 1):
        mid = movie["movie_id"]
        cache[mid] = {"hash": hashes[mid], "vector": embed_text(texts[mid])}
        if verbose:
            print(f"  [{i}/{len(stale)}] {movie['korean_title'][:20]}")

    cache = {mid: entry for mid, entry in cache.items() if mid in texts}  # DB에서 빠진 영화 정리
    write_json(MOVIE_EMBEDDING_PATH, {
        "model": EMBEDDING_REPO,
        "items": {str(k): v for k, v in cache.items()},
    })
    if verbose:
        print(f"[embedding] 캐시 저장 완료 -> {MOVIE_EMBEDDING_PATH}")

    return {mid: cache[mid]["vector"] for mid in texts if mid in cache}
