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
  - 영화 임베딩(자연어 취향 반영)은 기본 ON. 끄려면 환경변수 MVTI_USE_EMBEDDING=0.
    꺼져 있거나 로드에 실패해도 MVTI 4축 유사도만으로 추천은 정상 동작한다.
  - OPENAI_API_KEY가 있으면 유형 설명/추천 근거를 생성, 없으면 기본 문구로 대체.
"""

import os
import random
import unicodedata
from typing import Dict, Optional

from dotenv import find_dotenv, load_dotenv

# .env 로드: 프로젝트 루트(또는 그 상위)에 있는 OPENAI_API_KEY를 잡는다.
# override=True: OS 환경변수에 옛 키가 박혀 있어도 .env 값이 우선되게 한다.
load_dotenv(find_dotenv(usecwd=True), override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import AXIS_KEYS, FRONT_DIR, TOP_N
from dataset import load_movies, load_questions, load_types, read_json
from engine import build_user_profile, generate_result_copy, recommend

# ------------------------------------------------------------
# 시작 시 1회 로드 (데이터는 요청마다 다시 읽지 않는다)
# ------------------------------------------------------------

# 자연어 취향 임베딩 기본 ON (끄려면 MVTI_USE_EMBEDDING=0)
USE_EMBEDDING = os.environ.get("MVTI_USE_EMBEDDING", "1") != "0"


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

# ------------------------------------------------------------
# 스낵 / 특별관 추천용 데이터
#   - 이미지: FE/public/{snack,special_screen}/  (파일명 = 이름)
#   - 스낵은 이미지가 있는 것만 랜덤 후보로 사용
# ------------------------------------------------------------

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SNACK_IMG_DIR = os.path.join(_ROOT_DIR, "FE", "public", "snack")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _image_stems(folder: str):
    if not os.path.isdir(folder):
        return set()
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return {_nfc(os.path.splitext(f)[0]) for f in os.listdir(folder) if f.lower().endswith(exts)}


_snack_imgs = _image_stems(_SNACK_IMG_DIR)
_snack_db = read_json(os.path.join(FRONT_DIR, "cgv_snack_db.json"), default=[])
_SNACK_BY_NAME = {_nfc(s["name"]): s for s in _snack_db}


def _movie_goods_names(movies):
    """영화 snack 필드에 등장하는 스낵 = '굿즈'(미니언즈/모아나/포키/하츄핑 세트 등)."""
    names = set()
    for mv in movies:
        s = mv.get("snack")
        if not s:
            continue
        for x in (s if isinstance(s, list) else [s]):
            names.add(_nfc(x))
    return names


GOODS_NAMES = _movie_goods_names(MOVIES)
# 일반 스낵 랜덤 풀 = 이미지 있고 + 굿즈가 아닌 스낵
REGULAR_SNACKS = [
    s for s in _snack_db
    if _nfc(s["name"]) in _snack_imgs and _nfc(s["name"]) not in GOODS_NAMES
]
# 특별관 이름 -> 설명
SPECIAL_BY_NAME = {
    s["name"]: s for s in read_json(os.path.join(FRONT_DIR, "cgv_special_db.json"), default=[])
}


def _snack_card(entry: Dict) -> Dict:
    return {
        "kind": "snack",
        "name": entry["name"],
        "description": entry.get("description", ""),
        "price": entry.get("price"),
    }


def recommend_addons(movie: Dict):
    """
    영화마다 붙일 카드 2개(슬롯1, 슬롯2)를 반환.
      슬롯1 = 스낵: 굿즈 있으면 그 굿즈, 없으면 일반스낵 랜덤 1개
      슬롯2 = 상영관: 매칭 상영관 있으면 랜덤 1개, 없으면 일반스낵 랜덤 1개(슬롯1과 중복 X)
    """
    used = set()  # 이미 뽑은 스낵 이름(NFC) - 중복 방지

    # 슬롯1: 굿즈 or 일반스낵
    goods = movie.get("snack")
    gname = _nfc(goods[0] if isinstance(goods, list) else goods) if goods else None
    if gname and gname in _SNACK_BY_NAME:
        slot1 = _snack_card(_SNACK_BY_NAME[gname])
    else:
        pick = random.choice(REGULAR_SNACKS)
        slot1 = _snack_card(pick)
    used.add(_nfc(slot1["name"]))

    # 슬롯2: 상영관 or 일반스낵(중복 X)
    specials = [s for s in (movie.get("special_screen") or []) if s in SPECIAL_BY_NAME]
    if specials:
        name = random.choice(specials)
        slot2 = {"kind": "special", "name": name, "description": SPECIAL_BY_NAME[name].get("description", "")}
    else:
        pool = [s for s in REGULAR_SNACKS if _nfc(s["name"]) not in used] or REGULAR_SNACKS
        slot2 = _snack_card(random.choice(pool))

    return [slot1, slot2]

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

    movies_out = []
    for rec in recommendations:
        movies_out.append({
            **rec.as_dict(),
            "reason": reasons.get(rec.movie["movie_id"], ""),
            "addons": recommend_addons(rec.movie),  # 카드 2개 [슬롯1(스낵/굿즈), 슬롯2(상영관/스낵)]
        })

    return {
        **profile.as_dict(),
        "likes": type_info.get("likes", []),
        "dislikes": type_info.get("dislikes", []),
        "type_description": copy["type_description"],
        "movies": movies_out,
    }
