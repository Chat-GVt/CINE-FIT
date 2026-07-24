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

# B6: 자연어 입력이 있을 때는 임베딩(현재 취향 발화)에 더 무게를 둔다.
# 0.6:0.4 그대로면 축 유사도의 편차가 임베딩 편차를 덮어버려서 자연어가 결과에 거의 반영되지 않음.
NL_AXIS_WEIGHT = 0.3
NL_EMBED_WEIGHT = 0.7

# 사용자 임베딩을 구성하는 세 취향 신호의 상대 가중치.
# 좋아요(명시적 선호) > 자연어(현재 발화) > 관람내역(봤다는 사실, 호불호는 불명확) 순.
# 세 신호는 언제나 함께 결합된다(있는 것만). 관람내역은 재추천 제외와는 별개로 취향에도 반영.
LIKED_EMBED_WEIGHT = 1.0
NL_EMBED_SIGNAL_WEIGHT = 1.0
WATCHED_EMBED_WEIGHT = 0.3

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


def load_snacks() -> List[Dict]:
    return _load_json("cgv_snacks.json")["items"]


def load_goods() -> List[Dict]:
    return _load_json("cgv_goods.json")["items"]


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

def _avg_normalize(vecs: List[List[float]]) -> Optional[List[float]]:
    """벡터 리스트를 평균낸 뒤 L2 정규화. 비어 있으면 None."""
    if not vecs:
        return None
    dim = len(vecs[0])
    avg = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in avg))
    if norm == 0:
        return avg
    return [x / norm for x in avg]


def _movie_ids_to_vecs(movie_ids: List[int], movie_embeddings: Dict[int, List[float]]) -> List[List[float]]:
    return [movie_embeddings[mid] for mid in (movie_ids or []) if mid in movie_embeddings]


def user_embedding_from_likes(liked_movie_ids: List[int], movie_embeddings: Dict[int, List[float]]) -> List[float]:
    """좋아요한 영화들의 임베딩을 평균+정규화해서 '사용자 임베딩'으로 사용."""
    vec = _avg_normalize(_movie_ids_to_vecs(liked_movie_ids, movie_embeddings))
    if vec is None:
        return [0.0] * len(next(iter(movie_embeddings.values())))
    return vec


def build_user_embedding(
    movie_embeddings: Dict[int, List[float]],
    liked_movie_ids: Optional[List[int]] = None,
    watched_movie_ids: Optional[List[int]] = None,
    natural_language_text: Optional[str] = None,
    liked_weight: float = LIKED_EMBED_WEIGHT,
    watched_weight: float = WATCHED_EMBED_WEIGHT,
    nl_weight: float = NL_EMBED_SIGNAL_WEIGHT,
) -> List[float]:
    """좋아요 + 관람내역 + 자연어 발화를 가중 결합한 '사용자 임베딩'.

    - 세 신호 중 존재하는 것만 결합한다(각 신호는 먼저 평균+정규화한 뒤 가중합).
    - 좋아요는 자연어 입력 유무와 관계없이 항상 반영된다.
    - 관람내역은 취향 신호로 반영되지만, 재추천 제외(exclude_movie_ids)는 별도로 유지된다.
    - 결합 후 다시 L2 정규화해서 코사인 유사도에 바로 쓸 수 있게 한다.
    """
    components = []  # (weight, unit_vector)

    liked_vec = _avg_normalize(_movie_ids_to_vecs(liked_movie_ids, movie_embeddings))
    if liked_vec is not None:
        components.append((liked_weight, liked_vec))

    watched_vec = _avg_normalize(_movie_ids_to_vecs(watched_movie_ids, movie_embeddings))
    if watched_vec is not None:
        components.append((watched_weight, watched_vec))

    if natural_language_text:
        from bge_embedder import embed_text
        nl_vec = embed_text(natural_language_text)
        components.append((nl_weight, nl_vec))

    if not components:
        return [0.0] * len(next(iter(movie_embeddings.values())))

    dim = len(components[0][1])
    total_w = sum(w for w, _ in components) or 1.0
    combined = [sum(w * vec[i] for w, vec in components) / total_w for i in range(dim)]
    return _avg_normalize([combined]) or combined


def hybrid_movie_score(
    user_axis_vector: Dict[str, float],
    user_embedding: List[float],
    movie: Dict,
    movie_embeddings: Dict[int, List[float]],
    axis_weight: float = AXIS_WEIGHT,
    embed_weight: float = EMBED_WEIGHT,
) -> Dict[str, float]:
    """4축 유사도 + 임베딩 유사도 가중합(가중치 조절 가능). 세부 유사도도 같이 반환."""
    movie_axis_vector = centerize_vector(movie["mvti"]["axis_scores"])
    axis_sim = cosine_similarity(user_axis_vector, movie_axis_vector)

    movie_embed = movie_embeddings.get(movie["movie_id"])
    embed_sim = cosine_similarity_list(user_embedding, movie_embed) if movie_embed else 0.0

    hybrid = axis_weight * axis_sim + embed_weight * embed_sim
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
    axis_weight: float = AXIS_WEIGHT,
    embed_weight: float = EMBED_WEIGHT,
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

        scores = hybrid_movie_score(
            user_axis_vector, user_embedding, movie, movie_embeddings,
            axis_weight=axis_weight, embed_weight=embed_weight,
        )
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
    snacks: List[Dict],
    goods: List[Dict],
    top_n: int = 3,
    allow_alcohol: bool = False,
    natural_language_text: Optional[str] = None,
) -> List[Dict]:
    """
    persona.target_axis_scores(설문 없이 정답 벡터 그대로) + liked_movie_ids(임베딩 평균)를
    하이브리드 매칭에 넣어서 영화 추천 카드를 만든다.
    이미 시청한(watched_movie_ids) 영화는 추천 후보에서 제외.
    각 카드에 특별관/매점/굿즈까지 포함.

    임베딩(취향 벡터)은 좋아요 + 관람내역(+ 있다면 자연어)을 항상 함께 결합한다
    (build_user_embedding). 관람내역은 취향에는 반영되지만 watched_movie_ids로
    재추천에서는 계속 제외된다.

    natural_language_text(B6, 자연어 취향 입력)가 주어지면 추가로:
    - 임베딩: 위 결합에 이 문장의 BGE-M3 임베딩을 한 신호로 더 얹는다
    - 축: LLM이 문장에서 추정한 I/L/F/P를 target_axis_scores와 50:50으로 가중평균해서 사용
    - 축:임베딩 가중치도 0.3:0.7로 전환(자연어가 실제로 반영되도록)
    반환: [{title, poster_url, similarity, axis_similarity, embed_similarity, genre,
             screen, snacks, goods}, ...]
    """
    if natural_language_text:
        from llm_axis import convert_text_to_axis_scores

        nl_axis = convert_text_to_axis_scores(natural_language_text)
        blended_axis = {
            k: 0.5 * persona["target_axis_scores"][k] + 0.5 * nl_axis[k]
            for k in AXIS_KEYS
        }
        user_axis_vector = centerize_vector(blended_axis)
        axis_weight, embed_weight = NL_AXIS_WEIGHT, NL_EMBED_WEIGHT
    else:
        user_axis_vector = centerize_vector(persona["target_axis_scores"])
        axis_weight, embed_weight = AXIS_WEIGHT, EMBED_WEIGHT

    # 좋아요 + 관람내역 + (있다면)자연어를 항상 함께 결합한 사용자 임베딩.
    # 관람내역은 여기서 취향으로 반영되지만, 아래 recommend_movies_hybrid의
    # exclude_movie_ids로 재추천에서는 계속 제외된다.
    user_embedding = build_user_embedding(
        movie_embeddings,
        liked_movie_ids=persona["liked_movie_ids"],
        watched_movie_ids=persona["watched_movie_ids"],
        natural_language_text=natural_language_text,
    )

    top_movies = recommend_movies_hybrid(
        user_axis_vector, user_embedding, movies, movie_embeddings,
        exclude_movie_ids=persona["watched_movie_ids"],
        top_n=top_n,
        axis_weight=axis_weight,
        embed_weight=embed_weight,
    )

    cards = []
    for c in top_movies:
        movie = c["movie"]
        cards.append({
            "title": movie["korean_title"],
            "poster_url": movie["poster_url"],
            "similarity": c["similarity"],
            "axis_similarity": c["axis_similarity"],
            "embed_similarity": c["embed_similarity"],
            "genre": movie["genre"],
            "screen": recommend_screen(user_axis_vector, movie),
            "snacks": recommend_snacks(user_axis_vector, movie, snacks, allow_alcohol=allow_alcohol),
            "goods": recommend_goods(movie, goods),
        })
    return cards


# ---------- B7: 특별관 추천 ----------
# 취향 벡터로 추천하는 대상은 이 5종뿐. 나머지(SUITE CINEMA, CINE de CHEF, GOLD CLASS,
# 씨네앤포레, 씨네앤리빙룸)는 취향이 아니라 예산·상황으로 고르는 프리미엄관이라 제외.
SCREEN_AFFINITY = {
    "IMAX": {"I": 0.3, "L": 0.0, "F": 0.3, "P": 0.8},
    "SCREENX": {"I": 0.8, "L": 0.0, "F": 0.0, "P": 0.0},
    "4DX": {"I": 0.0, "L": 0.0, "F": 0.8, "P": 0.0},
    "ULTRA 4DX": {"I": 0.4, "L": 0.0, "F": 0.8, "P": 0.0},
    "DOLBY ATMOS": {"I": 0.0, "L": -0.8, "F": 0.0, "P": 0.0},
}
SCREEN_REASON = {
    "IMAX": "대중적이고 스케일 큰 취향과 잘 맞는 대형 스크린",
    "SCREENX": "상상력을 자극하는 파노라마 스크린",
    "4DX": "빠른 전개·액션감을 살리는 체감형 상영관",
    "ULTRA 4DX": "빠른 전개와 파노라마 몰입감을 동시에 주는 상영관",
    "DOLBY ATMOS": "어두운 무드를 사운드로 깊이 살리는 상영관",
}


def recommend_screen(user_axis_vector: Dict[str, float], movie: Dict) -> Optional[Dict]:
    candidates = [s for s in movie.get("cgv_screen_types", []) if s in SCREEN_AFFINITY]
    if not candidates:
        return None
    best_screen = max(candidates, key=lambda s: cosine_similarity(user_axis_vector, SCREEN_AFFINITY[s]))
    return {"screen": best_screen, "reason": SCREEN_REASON[best_screen]}


# ---------- B7: 매점 추천 ----------
COMBO_CATEGORIES = {"콤보", "세트"}


def _snack_hard_filter(snacks: List[Dict], movie_title: str, allow_alcohol: bool) -> List[Dict]:
    result = []
    for it in snacks:
        if it["category"] == "소스":
            continue
        if it["is_alcohol"] and not allow_alcohol:
            continue
        linked = it.get("linked_movie")
        if linked is not None and linked != movie_title:
            continue
        result.append(it)
    return result


def _snack_manner_filter(snacks: List[Dict], movie_axis_vector: Dict[str, float]) -> List[Dict]:
    f, p = movie_axis_vector["F"], movie_axis_vector["P"]
    result = []
    for it in snacks:
        if f < -0.6 and it["noise_level"] > 2:
            continue
        elif f < -0.3 and it["noise_level"] > 3:
            continue
        if p < -0.5 and (it["noise_level"] > 2 or it["smell_level"] > 2):
            continue
        result.append(it)
    return result


def recommend_snacks(
    user_axis_vector: Dict[str, float],
    movie: Dict,
    snacks: List[Dict],
    allow_alcohol: bool = False,
) -> List[Dict]:
    movie_title = movie["korean_title"]
    movie_axis_vector = centerize_vector(movie["mvti"]["axis_scores"])

    candidates = _snack_hard_filter(snacks, movie_title, allow_alcohol)
    candidates = _snack_manner_filter(candidates, movie_axis_vector)

    forced = [it for it in candidates if it.get("linked_movie") == movie_title]
    rest = [it for it in candidates if it.get("linked_movie") != movie_title]
    rest.sort(key=lambda it: cosine_similarity(user_axis_vector, it["pairing_vector"]), reverse=True)

    picked = []
    for it in forced:
        picked.append({"item_id": it["item_id"], "name": it["name"], "price": it["price"],
                        "reason": f"'{movie_title}' 연계 상품 우선 추천"})

    combo = next((it for it in rest if it["category"] in COMBO_CATEGORIES), None)
    if combo:
        sim = cosine_similarity(user_axis_vector, combo["pairing_vector"])
        picked.append({"item_id": combo["item_id"], "name": combo["name"], "price": combo["price"],
                        "reason": f"취향 일치도 {round((sim + 1) / 2 * 100, 1)}%"})
    else:
        drink = next((it for it in rest if it["category"] == "음료"), None)
        snack = next((it for it in rest if it["category"] not in COMBO_CATEGORIES | {"음료"}), None)
        for it in (drink, snack):
            if it:
                sim = cosine_similarity(user_axis_vector, it["pairing_vector"])
                picked.append({"item_id": it["item_id"], "name": it["name"], "price": it["price"],
                                "reason": f"취향 일치도 {round((sim + 1) / 2 * 100, 1)}%"})

    return picked


# ---------- B7: 굿즈 추천 ----------

def recommend_goods(movie: Dict, goods: List[Dict]) -> List[Dict]:
    """
    goods.linked_movie == movie.korean_title인 상품이 있으면 전부 반환.
    (movie_db_scored.json의 goods_package 플래그와 cgv_goods.json의 linked_movie 대상이
    서로 겹치지 않는 데이터 불일치가 있어 linked_movie 매칭만으로 판단)
    """
    movie_title = movie["korean_title"]
    return [
        {"item_id": g["item_id"], "name": g["name"], "price": g["price"]}
        for g in goods
        if g.get("linked_movie") == movie_title
    ]


if __name__ == "__main__":
    movies = load_movies()
    movie_embeddings = load_movie_embeddings()
    personas = load_personas()
    snacks = load_snacks()
    goods = load_goods()

    print(f"영화 {len(movies)}편, 임베딩 {len(movie_embeddings)}개, 매점 {len(snacks)}개, 굿즈 {len(goods)}개 로드")

    # STEP0 자체 테스트
    assert centerize(0) == -1.0 and centerize(100) == 1.0
    print("centerize OK")

    # B7 포함 자체 테스트: recommend_for_persona()로 특별관/매점/굿즈까지 연결됐는지 확인
    sample_ids = ["P-001", "P-022", "P-031", "P-040"]
    for pid in sample_ids:
        p = next(pp for pp in personas if pp["persona_id"] == pid)
        cards = recommend_for_persona(p, movies, movie_embeddings, snacks, goods)

        print(f"\n=== {p['persona_id']} {p['name']} ({p['target_type']}) ===")
        for c in cards:
            screen = c["screen"]["screen"] if c["screen"] else "없음(2D뿐)"
            snack_names = ", ".join(s["name"] for s in c["snacks"]) or "없음"
            goods_names = ", ".join(g["name"] for g in c["goods"]) or "없음"
            print(f"  [{c['similarity']}%] {c['title']} {c['genre']}")
            print(f"    특별관: {screen} | 매점: {snack_names} | 굿즈: {goods_names}")
