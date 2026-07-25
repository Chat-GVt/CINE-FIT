"""
추천 엔진.

  [1] 유사도 유틸    - centerize, 코사인 유사도, 0~100 변환
  [2] 사용자 MVTI    - 리커트 응답 -> 4축 점수 -> 유형 코드/이름/비율
  [3] 추천 점수      - 두 유사도 가중합 -> TOP N
  [4] 결과 문구      - OpenAI로 유형 설명 + 영화별 추천 근거 생성

최종 점수 = W_AXIS * (사용자 MVTI 4축 ↔ 영화 MVTI 4축 코사인 유사도)
          + W_EMBED * (자연어 취향 임베딩 ↔ 영화 줄거리·키워드 임베딩 코사인 유사도)
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    AXIS_KEYS,
    AXIS_POLES,
    LIKERT_MAX,
    LIKERT_MIN,
    OPENAI_API_KEY_ENV,
    OPENAI_MODEL,
    TOP_N,
    W_AXIS,
    W_EMBED,
)

# ============================================================
# [1] 유사도 유틸
# ============================================================


def centerize(score_0_100: float) -> float:
    """0~100 축 점수를 -1~1로 변환. 50(중립)이 0이 된다."""
    return (score_0_100 - 50) / 50


def centerize_axis(axis_scores: Dict[str, float]) -> Dict[str, float]:
    return {k: centerize(axis_scores[k]) for k in AXIS_KEYS if k in axis_scores}


def cosine_dict(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """공통 키만 사용하는 dict 벡터(4축) 코사인 유사도."""
    keys = [k for k in vec_a if k in vec_b]
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(vec_a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(vec_b[k] ** 2 for k in keys))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_list(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """리스트 벡터(임베딩) 코사인 유사도."""
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def to_percent(cosine_score: float) -> float:
    """-1~1 코사인 값을 0~100 표시용 점수로."""
    return round((cosine_score + 1) / 2 * 100, 1)


# ============================================================
# [2] 사용자 MVTI
# ============================================================
# questions.json의 각 문항은
#   axis      : I / L / F / P 중 하나
#   direction : +1이면 '동의할수록 high 극(I,L,F,P)', -1이면 역채점 문항


@dataclass
class UserProfile:
    """추천 엔진에 들어가는 사용자 입력 전체."""
    axis_scores: Dict[str, float]           # {"I": 0~100, ...}
    type_code: str                          # 예: "ILFP"
    type_name: str                          # 예: "화려한 마블 히어로형"
    characters: List[Dict]                  # 유형 대표 캐릭터 3인
    ratios: Dict[str, Dict]                 # 축별 우세 극과 비율
    natural_language: Optional[str] = None  # 자연어 취향 입력
    axis_vector: Dict[str, float] = field(init=False)  # -1~1 (유사도 계산용)

    def __post_init__(self):
        self.axis_vector = centerize_axis(self.axis_scores)

    @property
    def has_axis_signal(self) -> bool:
        """4축이 전부 50점이면 축 벡터가 영벡터라 모든 영화 유사도가 0이 된다. 그 경우 판별."""
        return any(abs(v) > 1e-9 for v in self.axis_vector.values())

    @property
    def keywords(self) -> List[str]:
        """["상상", "밝음", "빠름", "대중"] - 유형 화면 상단 표기용."""
        return [self.ratios[axis]["label"] for axis in AXIS_KEYS]

    def summary(self) -> str:
        parts = [f"{r['pole']}({r['label']}) {r['ratio']}%" for r in
                 (self.ratios[k] for k in AXIS_KEYS)]
        return f"{self.type_code} {self.type_name} | " + " · ".join(parts)

    def as_dict(self) -> Dict:
        return {
            "type_code": self.type_code,
            "type_name": self.type_name,
            "keywords": self.keywords,
            "characters": self.characters,
            "axis_scores": self.axis_scores,
            "ratios": self.ratios,
        }


def _validate_answers(questions: List[Dict], answers: Dict[str, int]) -> None:
    missing = [q["id"] for q in questions if q["id"] not in answers]
    if missing:
        raise ValueError(f"응답이 없는 문항: {missing}")

    out_of_range = [qid for qid, v in answers.items() if not (LIKERT_MIN <= v <= LIKERT_MAX)]
    if out_of_range:
        raise ValueError(f"{LIKERT_MIN}~{LIKERT_MAX} 범위를 벗어난 응답: {out_of_range}")


def compute_axis_scores(questions: List[Dict], answers: Dict[str, int]) -> Dict[str, float]:
    """축별 평균 응답을 0~100으로 환산. 역채점 문항은 (MIN+MAX-v)로 뒤집어 합산."""
    _validate_answers(questions, answers)

    grouped: Dict[str, List[float]] = {k: [] for k in AXIS_KEYS}
    for q in questions:
        value = answers[q["id"]]
        if q["direction"] == -1:
            value = LIKERT_MIN + LIKERT_MAX - value
        grouped[q["axis"]].append(value)

    scores = {}
    for axis, values in grouped.items():
        if not values:
            raise ValueError(f"{axis}축 문항이 없습니다.")
        mean = sum(values) / len(values)
        scores[axis] = round((mean - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN) * 100, 1)
    return scores


def axis_ratios(axis_scores: Dict[str, float]) -> Dict[str, Dict]:
    """
    축마다 우세한 극과 비율.
      I=90 -> {"pole": "I", "label": "상상", "ratio": 90.0}
      I=30 -> {"pole": "R", "label": "현실", "ratio": 70.0}
    """
    result = {}
    for axis in AXIS_KEYS:
        score = axis_scores[axis]
        if score >= 50:
            pole, label = AXIS_POLES[axis]["high"]
            ratio = score
        else:
            pole, label = AXIS_POLES[axis]["low"]
            ratio = 100 - score
        result[axis] = {"pole": pole, "label": label, "ratio": round(ratio, 1)}
    return result


def build_user_profile(
    questions: List[Dict],
    answers: Dict[str, int],
    types: Dict[str, Dict],
    natural_language: Optional[str] = None,
) -> UserProfile:
    """설문 응답 + 자연어 입력 -> UserProfile. types는 data.load_types() 결과."""
    axis_scores = compute_axis_scores(questions, answers)
    ratios = axis_ratios(axis_scores)
    type_code = "".join(ratios[axis]["pole"] for axis in AXIS_KEYS)

    if type_code not in types:
        raise ValueError(f"types.json에 없는 유형 코드입니다: {type_code}")

    return UserProfile(
        axis_scores=axis_scores,
        type_code=type_code,
        type_name=types[type_code]["name"],
        characters=types[type_code].get("characters", []),
        ratios=ratios,
        natural_language=(natural_language or "").strip() or None,
    )


# ============================================================
# [3] 추천 점수
# ============================================================


@dataclass
class Recommendation:
    """추천 결과 카드 1개."""
    movie: Dict
    total_score: float      # 0~100 최종 추천 점수
    axis_score: float       # 0~100 MVTI 축 유사도
    embed_score: float      # 0~100 텍스트 임베딩 유사도
    weights: Tuple[float, float]

    @property
    def title(self) -> str:
        return self.movie["korean_title"]

    def as_dict(self) -> Dict:
        return {
            "movie_id": self.movie["movie_id"],
            "title": self.title,
            "genre": self.movie.get("genre", []),
            "poster_url": self.movie.get("poster_url"),
            "total_score": self.total_score,
            "axis_score": self.axis_score,
            "embed_score": self.embed_score,
        }


def resolve_weights(has_embedding: bool, has_axis_signal: bool = True) -> Tuple[float, float]:
    """
    쓸 수 있는 신호에 따라 (축 가중치, 임베딩 가중치) 결정.
    - 자연어 입력이 없으면      -> 축 100%
    - 응답이 전부 중립(50점)이면 -> 임베딩 100%
    """
    if not has_embedding and not has_axis_signal:
        raise ValueError("축 응답도 중립이고 자연어 입력도 없어 추천 근거가 없습니다.")
    if not has_embedding:
        return 1.0, 0.0
    if not has_axis_signal:
        return 0.0, 1.0
    total = W_AXIS + W_EMBED
    return W_AXIS / total, W_EMBED / total


def score_movie(
    profile: UserProfile,
    user_embedding: Optional[List[float]],
    movie: Dict,
    movie_embeddings: Dict[int, List[float]],
    weights: Tuple[float, float],
) -> Recommendation:
    """영화 1편에 대한 두 유사도 계산 + 가중합."""
    w_axis, w_embed = weights

    # 1) MVTI 4축 유사도
    axis_sim = cosine_dict(profile.axis_vector, centerize_axis(movie["mvti"]["axis_scores"]))

    # 2) 텍스트 임베딩 유사도
    movie_embedding = movie_embeddings.get(movie["movie_id"])
    embed_sim = cosine_list(user_embedding, movie_embedding) if (user_embedding and movie_embedding) else 0.0

    return Recommendation(
        movie=movie,
        total_score=to_percent(w_axis * axis_sim + w_embed * embed_sim),
        axis_score=to_percent(axis_sim),
        embed_score=to_percent(embed_sim),
        weights=weights,
    )


def recommend(
    profile: UserProfile,
    movies: List[Dict],
    movie_embeddings: Dict[int, List[float]],
    user_embedding: Optional[List[float]] = None,
    top_n: int = TOP_N,
) -> List[Recommendation]:
    """영화 DB 전체를 점수화하고 상위 top_n편을 반환."""
    weights = resolve_weights(bool(user_embedding), profile.has_axis_signal)
    scored = [
        score_movie(profile, user_embedding, movie, movie_embeddings, weights)
        for movie in movies
    ]
    scored.sort(key=lambda r: r.total_score, reverse=True)
    return scored[:top_n]


# ============================================================
# [4] 결과 문구 생성 (OpenAI)
# ============================================================
# 유형 설명 + 영화별 추천 근거를 한 번의 호출로 같이 받는다.
# 따로 부르면 비용/지연이 2배가 되고, 유형 설명과 추천 이유의 톤이 서로 따로 논다.
# 사용 전: export OPENAI_API_KEY="sk-..."

COPY_SYSTEM_PROMPT = """너는 CGV CINE:FIT의 영화 취향 분석 카피라이터다.
사용자의 MVTI 취향 유형 결과 화면에 들어갈 문구를 쓴다.

## type_description (유형 설명)
- 3~4문장, 한국어 존댓말.
- 유형 이름의 이미지를 살려서 "어떤 영화를 즐기는 사람인지"를 설명한다.
- 축 비율이 65% 미만인 축은 단정하지 말고 "상황에 따라 오간다"는 뉘앙스로 쓴다.
- 대표 캐릭터는 1~2명만 자연스럽게 언급한다. 셋 다 나열하지 않는다.
- 자연어 취향 입력이 있으면 그 내용을 한 번 반영한다.

## reasons (영화별 추천 근거)
- 영화마다 2문장 이내, 한국어 존댓말.
- 반드시 (1) 사용자 취향 근거와 (2) 영화 특성 근거를 각각 한 번씩 언급한다.

## 공통 규칙
- 주어진 데이터에 없는 사실을 지어내지 않는다. 결말 스포일러 금지.
- 성격이나 인성을 규정하지 않는다("당신은 ~한 사람입니다" 금지). 취향 서술에 집중한다.
- 출력은 아래 JSON 형식만. 다른 텍스트나 마크다운 금지.

{"type_description": "...", "reasons": [{"movie_id": 1, "reason": "..."}]}"""


def build_copy_prompt(profile: UserProfile, recommendations: List[Recommendation]) -> str:
    characters = ", ".join(f"{c['name']}({c['work']})" for c in profile.characters)
    user_lines = [
        f"MVTI 유형: {profile.type_code} - {profile.type_name}",
        f"대표 캐릭터: {characters}",
    ]
    for axis in AXIS_KEYS:
        r = profile.ratios[axis]
        user_lines.append(f"- {axis}축: {r['label']}({r['pole']}) {r['ratio']}%")
    if profile.natural_language:
        user_lines.append(f"자연어로 밝힌 취향: {profile.natural_language}")

    movie_blocks = []
    for rec in recommendations:
        movie = rec.movie
        axis_scores = movie["mvti"]["axis_scores"]
        axis_reasons = movie["mvti"].get("reasons", {})

        lines = [
            f"[movie_id={movie['movie_id']}] {movie['korean_title']}",
            f"- 장르: {', '.join(movie.get('genre', []))}",
            f"- 줄거리: {movie.get('overview', '')}",
            "- MVTI 점수: " + ", ".join(f"{k}={axis_scores[k]}" for k in AXIS_KEYS),
        ]
        lines += [f"  · {a}축 근거: {axis_reasons[a]}" for a in AXIS_KEYS if a in axis_reasons]
        lines.append(
            f"- 매칭 점수: 종합 {rec.total_score}점 "
            f"(취향유형 {rec.axis_score} / 자연어취향 {rec.embed_score})"
        )
        movie_blocks.append("\n".join(lines))

    return (
        "## 사용자\n" + "\n".join(user_lines) + "\n\n"
        "## 추천된 영화\n" + "\n\n".join(movie_blocks) + "\n\n"
        f"유형 설명 1개와 위 {len(recommendations)}편의 추천 근거를 작성해줘."
    )


def _fallback_type_description(profile: UserProfile) -> str:
    """API 키가 없거나 호출에 실패했을 때 쓰는 기본 유형 설명."""
    keywords = " · ".join(profile.keywords)
    names = ", ".join(c["name"] for c in profile.characters[:2])
    return f"{keywords} 취향의 {profile.type_name}이에요. {names} 같은 인물이 나오는 영화와 잘 맞아요."


def _fallback_reason(profile: UserProfile, rec: Recommendation) -> str:
    """API 키가 없거나 호출에 실패했을 때 쓰는 기본 추천 근거."""
    top_axis = max(AXIS_KEYS, key=lambda a: profile.ratios[a]["ratio"])
    label = profile.ratios[top_axis]["label"]
    genre = ", ".join(rec.movie.get("genre", [])[:2])
    return f"{label} 취향과 잘 맞는 {genre} 작품이에요."


def _fallback_copy(profile: UserProfile, recommendations: List[Recommendation]) -> Dict:
    return {
        "type_description": _fallback_type_description(profile),
        "reasons": {r.movie["movie_id"]: _fallback_reason(profile, r) for r in recommendations},
    }


def generate_result_copy(
    profile: UserProfile,
    recommendations: List[Recommendation],
) -> Dict:
    """
    {"type_description": str, "reasons": {movie_id: str}} 반환.
    호출에 실패해도 기본 문구로 대체해서 결과 화면 자체는 항상 뜨게 한다.
    """
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        print(f"[copy] {OPENAI_API_KEY_ENV}가 없어 기본 문구로 대체합니다.")
        return _fallback_copy(profile, recommendations)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.8,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": COPY_SYSTEM_PROMPT},
                {"role": "user", "content": build_copy_prompt(profile, recommendations)},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        result = {
            "type_description": payload["type_description"],
            "reasons": {int(item["movie_id"]): item["reason"] for item in payload["reasons"]},
        }
    except Exception as e:  # 네트워크/파싱/쿼터 등 어떤 실패든
        print(f"[copy] 생성 실패({e}) - 기본 문구로 대체합니다.")
        return _fallback_copy(profile, recommendations)

    for rec in recommendations:  # 혹시 빠진 영화가 있으면 기본 문구로 채움
        result["reasons"].setdefault(rec.movie["movie_id"], _fallback_reason(profile, rec))
    return result
