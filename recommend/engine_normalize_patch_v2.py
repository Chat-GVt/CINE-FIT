"""
engine.py 정규화 패치

score_movie()와 recommend()를 아래 내용으로 교체하고, _zscore()/to_display()를 추가한다.

바뀌는 것:
  두 유사도를 원본 코사인 그대로 더하던 것을 → 각각 정규화한 뒤 가중합한다.
  축 유사도(표준편차 ~0.49)와 임베딩 유사도(~0.04)의 분산이 11배 넘게 차이 나서,
  정규화 없이는 가중치를 어떻게 주든 축이 순위를 독점하기 때문.

구조가 바뀌는 이유:
  정규화는 후보 전체의 분포를 알아야 계산할 수 있다.
  따라서 '영화 1편씩 점수화'하던 score_movie()는 더 이상 쓸 수 없고,
  recommend()가 전체 유사도를 모아 정규화한 뒤 결합한다.

주의:
  정규화 방식은 weight_grid_norm.py의 실험과 동일해야 한다(페르소나별 z-score).
  다른 방식을 쓰면 W_AXIS=0.9라는 실험 결과가 그대로 적용되지 않는다.
"""

from typing import Dict, List, Optional, Tuple


# ============================================================
# [1] 추가할 유틸  (to_percent 근처에 두면 됨)
# ============================================================


def _zscore(values: List[float]) -> List[float]:
    """평균 0, 표준편차 1로 정규화. 분산이 0이면(신호 없음) 전부 0."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    if sd < 1e-9:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def to_display(z: float) -> float:
    """
    정규화 점수(z)를 0~100 표시용 점수로.
    z=+3 -> 95점, z=0 -> 50점, z=-3 -> 5점.

    주의: 정규화 후 점수는 '후보군 안에서의 상대적 위치'를 의미한다.
         상영작 목록이 바뀌면 같은 영화라도 점수가 달라질 수 있다.
    """
    return round(max(0.0, min(100.0, 50.0 + 15.0 * z)), 1)


# ============================================================
# [2] recommend() 교체  (score_movie()는 삭제)
# ============================================================


def recommend(
    profile: "UserProfile",
    movies: List[Dict],
    movie_embeddings: Dict[int, List[float]],
    user_embedding: Optional[List[float]] = None,
    top_n: int = TOP_N,
) -> List["Recommendation"]:
    """
    영화 DB 전체를 점수화하고 상위 top_n편을 반환.

    1) 후보 전체의 두 유사도를 각각 수집
    2) 각각 z-score 정규화해 영향력을 맞춤
    3) 가중합 -> 정렬 -> 상위 N편
    """
    weights = resolve_weights(bool(user_embedding), profile.has_axis_signal)
    w_axis, w_embed = weights

    # 1) 원본 유사도 수집
    axis_sims: List[float] = []
    embed_sims: List[float] = []
    for movie in movies:
        axis_sims.append(
            cosine_dict(profile.axis_vector, centerize_axis(movie["mvti"]["axis_scores"]))
        )
        movie_embedding = movie_embeddings.get(movie["movie_id"])
        embed_sims.append(
            cosine_list(user_embedding, movie_embedding)
            if (user_embedding and movie_embedding) else 0.0
        )

    # 2) 정규화 (분산 차이 보정)
    z_axis = _zscore(axis_sims)
    z_embed = _zscore(embed_sims)

    # 3) 가중합
    scored: List[Tuple[float, "Recommendation"]] = []
    for i, movie in enumerate(movies):
        blended = w_axis * z_axis[i] + w_embed * z_embed[i]
        scored.append((
            blended,
            Recommendation(
                movie=movie,
                total_score=to_display(blended),
                axis_score=to_percent(axis_sims[i]),    # 세부 점수는 원본 코사인 기준 유지
                embed_score=to_percent(embed_sims[i]),  # (사용자가 해석 가능한 값)
                weights=weights,
            ),
        ))

    # 표시 점수는 반올림되므로, 정렬은 반올림 전 값으로 해야 순위가 뭉개지지 않는다
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [rec for _, rec in scored[:top_n]]


# ============================================================
# [3] 함께 바꿀 것
# ============================================================
#
# config.py
#   W_AXIS  = 0.9
#   W_EMBED = 0.1
#
# main.py 61번째 줄 근처
#   print("\nBGE-M3로 자연어 취향 임베딩 중...")
#   -> 실제 모델은 KURE-v1이므로 문구 수정
#
# 적용 후 pipeline_test.py를 다시 돌려 최종 수치를 갱신할 것.
