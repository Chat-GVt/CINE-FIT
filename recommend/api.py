"""
CINE:FIT MVTI 결과 API (FastAPI).

프론트(UI/mvti-test-new.jsx)가 설문 결과 화면을 그릴 때 호출한다.

  POST /mvti/score
    요청  { "answers": {"Q01": 5, ...}, "natural_language": "..." (선택) }
    응답  { type_code, type_name, keywords, axis_scores, ratios,
            characters, type_description, movies:[...] }

실행:
  cd recommend
  uvicorn api:app --reload --port 8000

동작 원리:
  - MVTI 계산 / 유형 판정 / 추천 / 문구 생성은 전부 engine.py의 기존 함수를 그대로 쓴다.
  - 영화 임베딩(자연어 취향 반영)은 무거워서 기본 비활성. 켜려면 환경변수
    MVTI_USE_EMBEDDING=1. 꺼져 있어도 MVTI 4축 유사도만으로 추천은 정상 동작한다.
  - OPENAI_API_KEY가 있으면 유형 설명/추천 근거를 생성, 없으면 기본 문구로 대체.
"""

import os
from typing import Dict, Optional

from dotenv import find_dotenv, load_dotenv

# .env 로드: 프로젝트 루트(또는 그 상위)에 있는 OPENAI_API_KEY를 잡는다.
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import AXIS_KEYS, TOP_N
from dataset import load_movies, load_questions, load_types
from engine import build_user_profile, generate_result_copy, recommend

# ------------------------------------------------------------
# 시작 시 1회 로드 (데이터는 요청마다 다시 읽지 않는다)
# ------------------------------------------------------------

USE_EMBEDDING = os.environ.get("MVTI_USE_EMBEDDING") == "1"


def _sanitize_movies(movies):
    """
    영화 축 점수에 null이 섞여 있으면(스코어링 미완료) 유사도 계산이 터진다.
    빠진 축은 50(중립)으로 채워서 그 축이 추천에 영향을 주지 않게 한다.
    """
    for m in movies:
        scores = m.get("mvti", {}).get("axis_scores", {})
        for k in AXIS_KEYS:
            if scores.get(k) is None:
                scores[k] = 50
        m.setdefault("mvti", {})["axis_scores"] = scores
    return movies


MOVIES = _sanitize_movies(load_movies())
QUESTIONS = load_questions()
TYPES = load_types()

# 임베딩은 옵션. 실패해도 서버는 뜨고, 축 유사도만으로 추천한다.
MOVIE_EMBEDDINGS: Dict[int, list] = {}
if USE_EMBEDDING:
    try:
        from dataset import get_movie_embeddings

        MOVIE_EMBEDDINGS = get_movie_embeddings(MOVIES)
    except Exception as e:  # noqa: BLE001
        print(f"[api] 임베딩 로드 실패({e}) - 축 유사도만으로 추천합니다.")
        MOVIE_EMBEDDINGS = {}


# ------------------------------------------------------------
# 요청/응답 스키마
# ------------------------------------------------------------


class ScoreRequest(BaseModel):
    answers: Dict[str, int]
    natural_language: Optional[str] = None


# ------------------------------------------------------------
# 앱
# ------------------------------------------------------------

app = FastAPI(title="CINE:FIT MVTI API")

# 개발 편의상 모든 origin 허용 (배포 시 도메인 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "movies": len(MOVIES),
        "questions": len(QUESTIONS),
        "types": len(TYPES),
        "use_embedding": USE_EMBEDDING,
        "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.post("/mvti/score")
def score(req: ScoreRequest):
    """설문 응답 -> MVTI 유형 + 게이지 비율 + 추천 문구 + 영화 TOP N."""
    try:
        profile = build_user_profile(
            questions=QUESTIONS,
            answers=req.answers,
            types=TYPES,
            natural_language=req.natural_language,
        )
    except ValueError as e:  # 응답 누락/범위 오류 등
        raise HTTPException(status_code=400, detail=str(e))

    # 자연어 취향 임베딩 (옵션)
    user_embedding = None
    if USE_EMBEDDING and profile.natural_language:
        try:
            from dataset import embed_text

            user_embedding = embed_text(profile.natural_language)
        except Exception as e:  # noqa: BLE001
            print(f"[api] 자연어 임베딩 실패({e}) - 축 유사도만 사용합니다.")

    recommendations = recommend(
        profile=profile,
        movies=MOVIES,
        movie_embeddings=MOVIE_EMBEDDINGS,
        user_embedding=user_embedding,
        top_n=TOP_N,
    )

    copy = generate_result_copy(profile, recommendations)
    reasons = copy["reasons"]

    # 유형별 좋아하는/싫어하는 영화 특징 (types.json에 있으면 함께 반환)
    type_info = TYPES.get(profile.type_code, {})

    return {
        **profile.as_dict(),
        "likes": type_info.get("likes", []),
        "dislikes": type_info.get("dislikes", []),
        "type_description": copy["type_description"],
        "movies": [
            {**rec.as_dict(), "reason": reasons.get(rec.movie["movie_id"], "")}
            for rec in recommendations
        ],
    }
