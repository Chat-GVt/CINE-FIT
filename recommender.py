"""
CINE:FIT 추천 엔진 (하이브리드: 4축 MVTI + BGE-M3 임베딩)
"""

import json
import math
import os
from typing import Dict, List, Optional

AXIS_KEYS = ("I", "L", "F", "P")
AXIS_WEIGHT = 0.6
EMBED_WEIGHT = 0.4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(filename: str):
    path = os.path.join(BASE_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_movies() -> List[Dict]:
    return _load_json("movie_db_scored.json")


def load_movie_embeddings() -> Dict[int, List[float]]:
    raw = _load_json("movie_embeddings.json")
    return {int(k): v for k, v in raw.items()}


def load_personas() -> List[Dict]:
    return _load_json("personas.json")["personas"]


# ---------- STEP 0: 공통 유틸 ----------

def centerize(score_0_100: float) -> float:
    return (score_0_100 - 50) / 50


def centerize_vector(axis_scores_0_100: Dict[str, float]) -> Dict[str, float]:
    return {k: centerize(axis_scores_0_100[k]) for k in AXIS_KEYS if k in axis_scores_0_100}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    keys = [k for k in vec_a if k in vec_b]
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(vec_a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(vec_b[k] ** 2 for k in keys))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_list(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------- B3: 하이브리드 매칭 ----------

def user_embedding_from_likes(liked_movie_ids: List[int], movie_embeddings: Dict[int, List[float]]) -> List[float]:
    """좋아요한 영화들의 임베딩을 평균+정규화해서 '사용자 임베딩'으로 사용."""
    vecs = [movie_embeddings[mid] for mid in liked_movie_ids if mid in movie_embeddings]
    if not vecs:
        return [0.0] * len(next(iter(movie_embeddings.values())))

    dim = len(vecs[0])
    avg = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in avg))
    if norm == 0:
        return avg
    return [x / norm for x in avg]


def hybrid_movie_score(
    user_axis_vector: Dict[str, float],
    user_embedding: List[float],
    movie: Dict,
    movie_embeddings: Dict[int, List[float]],
) -> Dict[str, float]:
    """4축 유사도(0.6) + 임베딩 유사도(0.4) 가중합. 세부 유사도도 같이 반환."""
    movie_axis_vector = centerize_vector(movie["mvti"]["axis_scores"])
    axis_sim = cosine_similarity(user_axis_vector, movie_axis_vector)

    movie_embed = movie_embeddings.get(movie["movie_id"])
    embed_sim = cosine_similarity_list(user_embedding, movie_embed) if movie_embed else 0.0

    hybrid = AXIS_WEIGHT * axis_sim + EMBED_WEIGHT * embed_sim
    return {"hybrid": hybrid, "axis_sim": axis_sim, "embed_sim": embed_sim}


def recommend_movies_hybrid(
    user_axis_vector: Dict[str, float],
    user_embedding: List[float],
    movies: List[Dict],
    movie_embeddings: Dict[int, List[float]],
    status_filter: Optional[List[str]] = None,
    exclude_documentary: bool = True,
    exclude_movie_ids: Optional[List[int]] = None,
    top_n: int = 3,
) -> List[Dict]:
    exclude_set = set(exclude_movie_ids or [])
    candidates = []
    for movie in movies:
        if movie["movie_id"] in exclude_set:
            continue
        if exclude_documentary and "다큐멘터리" in movie.get("genre", []):
            continue
        if status_filter is not None and movie.get("status") not in status_filter:
            continue

        scores = hybrid_movie_score(user_axis_vector, user_embedding, movie, movie_embeddings)
        candidates.append({
            "movie": movie,
            "similarity": round((scores["hybrid"] + 1) / 2 * 100, 1),
            "axis_similarity": round((scores["axis_sim"] + 1) / 2 * 100, 1),
            "embed_similarity": round((scores["embed_sim"] + 1) / 2 * 100, 1),
        })

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates[:top_n]


# ---------- B4: 페르소나 벡터 주입 -> 추천 결과 출력 연결 ----------

def recommend_for_persona(
    persona: Dict,
    movies: List[Dict],
    movie_embeddings: Dict[int, List[float]],
    top_n: int = 3,
) -> List[Dict]:
    """
    persona.target_axis_scores(설문 없이 정답 벡터 그대로) + liked_movie_ids(임베딩 평균)를
    하이브리드 매칭에 넣어서 영화 추천 카드를 만든다.
    이미 시청한(watched_movie_ids) 영화는 추천 후보에서 제외.
    반환: [{title, poster_url, similarity, axis_similarity, embed_similarity, genre}, ...]
    """
    user_axis_vector = centerize_vector(persona["target_axis_scores"])
    user_embedding = user_embedding_from_likes(persona["liked_movie_ids"], movie_embeddings)

    top_movies = recommend_movies_hybrid(
        user_axis_vector, user_embedding, movies, movie_embeddings,
        exclude_movie_ids=persona["watched_movie_ids"],
        top_n=top_n,
    )

    return [
        {
            "title": c["movie"]["korean_title"],
            "poster_url": c["movie"]["poster_url"],
            "similarity": c["similarity"],
            "axis_similarity": c["axis_similarity"],
            "embed_similarity": c["embed_similarity"],
            "genre": c["movie"]["genre"],
        }
        for c in top_movies
    ]


if __name__ == "__main__":
    movies = load_movies()
    movie_embeddings = load_movie_embeddings()
    personas = load_personas()

    print(f"영화 {len(movies)}편, 임베딩 {len(movie_embeddings)}개 로드")

    # STEP0 자체 테스트
    assert centerize(0) == -1.0 and centerize(100) == 1.0
    print("centerize OK")

    # B4 자체 테스트: recommend_for_persona()로 깔끔하게 연결됐는지 확인
    sample_ids = ["P-001", "P-022", "P-031", "P-040"]
    for pid in sample_ids:
        p = next(pp for pp in personas if pp["persona_id"] == pid)
        cards = recommend_for_persona(p, movies, movie_embeddings)

        print(f"\n=== {p['persona_id']} {p['name']} ({p['target_type']}) ===")
        for c in cards:
            print(f"  [{c['similarity']}% = axis {c['axis_similarity']}% / embed {c['embed_similarity']}%] "
                  f"{c['title']} {c['genre']}")
