"""
CINE:FIT 추천 API (FastAPI)

recommender.py의 순수 함수들을 REST 엔드포인트로 감싼 계층.
프론트엔드는 이 API만 호출하면 된다.

- 데이터(영화/임베딩/매점/굿즈/페르소나)는 서버 시작 시 1회 로딩해 재사용한다.
- BGE-M3 임베딩 모델과 LLM(자연어->축)은 무겁고 외부 의존이라, 자연어 입력이
  들어온 요청에서만 지연 로딩된다(recommender 내부에서 lazy import).
  * 자연어 입력에는 ANTHROPIC_API_KEY 환경변수가 필요하다.

실행:  uvicorn api:app --reload --port 8000
문서:  http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import llm_axis
import llm_personalize
import mvti
import mvti_questions
import recommender

AXIS_KEYS = recommender.AXIS_KEYS  # ("I", "L", "F", "P")

# 서버 생애 동안 재사용하는 로딩된 데이터 컨테이너
STATE: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작: JSON 데이터 1회 로딩 (임베딩 모델은 자연어 요청 시 지연 로딩)
    STATE["movies"] = recommender.load_movies()
    STATE["movie_embeddings"] = recommender.load_movie_embeddings()
    STATE["snacks"] = recommender.load_snacks()
    STATE["goods"] = recommender.load_goods()
    STATE["personas"] = {p["persona_id"]: p for p in recommender.load_personas()}
    yield
    STATE.clear()


app = FastAPI(
    title="CINE:FIT Recommendation API",
    version="1.0.0",
    description="MVTI 4축 + BGE-M3 임베딩 하이브리드 영화·상영관·매점·굿즈 추천",
    lifespan=lifespan,
)

# 프론트엔드(다른 오리진)에서 호출할 수 있도록 CORS 허용.
# 배포 시 allow_origins를 실제 프론트 도메인으로 좁히는 것을 권장.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 요청/응답 스키마 ----------

class RecommendRequest(BaseModel):
    mvti_scores: Dict[str, float] = Field(
        ..., description="MVTI 4축 점수(0~100). I/L/F/P 키 모두 필요.",
        json_schema_extra={"example": {"I": 70, "L": 60, "F": 55, "P": 80}},
    )
    liked_movie_ids: List[int] = Field(default_factory=list, description="좋아요한 영화 movie_id 목록")
    watched_movie_ids: List[int] = Field(default_factory=list, description="이미 본 영화 movie_id 목록(재추천 제외 + 취향 반영)")
    natural_language_text: Optional[str] = Field(default=None, description="자유 취향 문장(선택). 넣으면 임베딩·축에 함께 반영.")
    top_n: int = Field(default=3, ge=1, le=20)
    allow_alcohol: bool = Field(default=False, description="매점 추천에 주류 허용 여부")


class PersonaRecommendRequest(BaseModel):
    natural_language_text: Optional[str] = None
    top_n: int = Field(default=3, ge=1, le=20)
    allow_alcohol: bool = False


def _llm_call(fn, *args, **kwargs):
    """LLM 의존 함수 호출 래퍼. 키 누락/호출 실패를 사용자 친화적 HTTP 오류로 변환."""
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        if "ANTHROPIC_API_KEY" in str(e):
            raise HTTPException(
                status_code=503,
                detail="이 기능(LLM)에는 ANTHROPIC_API_KEY 환경변수가 필요합니다.",
            )
        raise
    except anthropic.APIError as e:  # 인증 실패·네트워크·모델 오류 등
        raise HTTPException(status_code=502, detail=f"LLM 호출 실패: {e}")


def _recommend(persona: Dict, top_n: int, allow_alcohol: bool, nl_text: Optional[str]) -> List[Dict]:
    """recommend_for_persona 호출 래퍼. 자연어 경로의 외부 의존 오류를 사용자 친화적으로 변환."""
    return _llm_call(
        recommender.recommend_for_persona,
        persona,
        STATE["movies"],
        STATE["movie_embeddings"],
        STATE["snacks"],
        STATE["goods"],
        top_n=top_n,
        allow_alcohol=allow_alcohol,
        natural_language_text=nl_text,
    )


def _type_name(target_type: str) -> Optional[str]:
    return mvti.TYPE_INFO.get(target_type, {}).get("name")


# ---------- 입력 단계 스키마 ----------

class SurveyRequest(BaseModel):
    answers: Dict[int, int] = Field(
        ..., description="문항번호(1~20) -> 응답값(1~5)",
        json_schema_extra={"example": {str(i): 3 for i in range(1, 21)}},
    )


class TextAxisRequest(BaseModel):
    text: str = Field(..., description="자유 취향 문장", min_length=1)


class CurationRequest(BaseModel):
    target_type: str = Field(..., description="MVTI 4글자 유형 코드(예: ILFP)")
    mvti_scores: Dict[str, float] = Field(..., description="I/L/F/P 0~100")
    cards: List[Dict] = Field(..., description="/recommend 가 반환한 카드 목록(top3)")
    natural_language_text: Optional[str] = None


class TypeIntroRequest(BaseModel):
    name: str
    target_type: str = Field(..., description="MVTI 4글자 유형 코드")
    natural_pref: str = Field(..., description="사용자 취향 메모(자유 입력)")


# ---------- 엔드포인트 ----------

@app.get("/health")
def health():
    """서버 상태 및 로딩된 데이터 규모."""
    return {
        "status": "ok",
        "movies": len(STATE.get("movies", [])),
        "embeddings": len(STATE.get("movie_embeddings", {})),
        "snacks": len(STATE.get("snacks", [])),
        "goods": len(STATE.get("goods", [])),
        "personas": len(STATE.get("personas", {})),
    }


@app.get("/personas")
def list_personas():
    """데모용 페르소나 요약 목록."""
    return [
        {
            "persona_id": p["persona_id"],
            "name": p.get("name"),
            "target_type": p.get("target_type"),
        }
        for p in STATE["personas"].values()
    ]


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    개별 취향 필드로 영화·상영관·매점·굿즈 추천.

    반환: { count, cards: [{ title, poster_url, similarity, axis_similarity,
            embed_similarity, genre, screen, snacks, goods }, ...] }
    """
    missing = [k for k in AXIS_KEYS if k not in req.mvti_scores]
    if missing:
        raise HTTPException(status_code=422, detail=f"mvti_scores에 {missing} 축이 없습니다. I/L/F/P 모두 필요합니다.")

    persona = {
        "target_axis_scores": {k: float(req.mvti_scores[k]) for k in AXIS_KEYS},
        "liked_movie_ids": req.liked_movie_ids,
        "watched_movie_ids": req.watched_movie_ids,
    }
    cards = _recommend(persona, req.top_n, req.allow_alcohol, req.natural_language_text)
    return {"count": len(cards), "cards": cards}


@app.post("/recommend/persona/{persona_id}")
def recommend_persona(persona_id: str, req: Optional[PersonaRecommendRequest] = None):
    """
    저장된 페르소나(persona_id) 기반 추천(데모용).
    body는 선택이며, natural_language_text를 주면 그 발화까지 함께 반영한다.
    """
    persona = STATE["personas"].get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"persona_id '{persona_id}' 를 찾을 수 없습니다.")

    req = req or PersonaRecommendRequest()
    cards = _recommend(persona, req.top_n, req.allow_alcohol, req.natural_language_text)
    return {
        "persona_id": persona_id,
        "name": persona.get("name"),
        "target_type": persona.get("target_type"),
        "count": len(cards),
        "cards": cards,
    }


# ---------- 입력 단계: MVTI 설문 ----------

@app.get("/mvti/questions")
def mvti_questions_list():
    """MVTI 20문항 텍스트. 프론트가 설문 화면을 그릴 때 사용(응답은 1~5 척도)."""
    return {
        "scale": {"min": 1, "max": 5, "hint": "1=전혀 아니다 ~ 5=매우 그렇다"},
        "questions": [
            {"no": no, "text": text}
            for no, text in sorted(mvti_questions.QUESTIONS.items())
        ],
    }


@app.post("/mvti/score")
def mvti_score(req: SurveyRequest):
    """
    설문 응답(1~20 문항, 각 1~5) -> 유형 코드 + 축 점수 + 유형 정보.
    반환: { type_code, axis_scores, type_name, keywords, characters }
    """
    try:
        result = mvti.get_survey_result(req.answers)
    except ValueError as e:  # 문항 누락 / 범위 초과
        raise HTTPException(status_code=422, detail=str(e))

    # /recommend, /curation, /type-intro 입력으로 바로 쓸 수 있는 평평한 축 점수도 함께 제공.
    result["axis_scores_flat"] = {
        cfg["primary"]: result["axis_scores"][axis_name][cfg["primary"]]
        for axis_name, cfg in mvti.AXES.items()
    }
    return result


# ---------- 입력 단계: 자연어 -> MVTI 축 (LLM) ----------

@app.post("/axis/from-text")
def axis_from_text(req: TextAxisRequest):
    """
    자유 취향 문장 -> 추정 MVTI 축(I/L/F/P 0~100) + 축별 근거.
    (프론트에서 추천 전에 '분석 결과 미리보기'로 보여주기 좋음)
    """
    return _llm_call(llm_axis.convert_text_to_axis_scores, req.text)


# ---------- 출력 문장 생성: 큐레이션 이유 / 유형 소개글 (LLM) ----------

@app.post("/curation")
def curation(req: CurationRequest):
    """
    추천된 3편 카드를 '묶음 단위'로 왜 골랐는지 설명하는 큐레이션 문장 생성.
    보통 /recommend 결과의 cards를 그대로 넘긴다.
    """
    missing = [k for k in AXIS_KEYS if k not in req.mvti_scores]
    if missing:
        raise HTTPException(status_code=422, detail=f"mvti_scores에 {missing} 축이 없습니다.")

    persona = {
        "target_type": req.target_type,
        "type_name": _type_name(req.target_type),
        "target_axis_scores": {k: float(req.mvti_scores[k]) for k in AXIS_KEYS},
    }
    reason = _llm_call(
        llm_personalize.generate_curation_reason,
        persona, req.cards, STATE["movies"],
        natural_language_text=req.natural_language_text,
    )
    return {"curation_reason": reason}


@app.post("/recommend/reasons")
def recommend_reasons(req: CurationRequest):
    """
    추천된 카드 각각에 붙는 '영화별 개인화 추천 이유' 생성(LLM, 한 번 호출로 전체).
    보통 /recommend 결과의 cards를 그대로 넘긴다.
    반환: { reasons: [{ title, reason }, ...] } (카드 순서 유지)
    """
    missing = [k for k in AXIS_KEYS if k not in req.mvti_scores]
    if missing:
        raise HTTPException(status_code=422, detail=f"mvti_scores에 {missing} 축이 없습니다.")

    persona = {
        "target_type": req.target_type,
        "type_name": _type_name(req.target_type),
        "target_axis_scores": {k: float(req.mvti_scores[k]) for k in AXIS_KEYS},
    }
    reasons = _llm_call(
        llm_personalize.generate_movie_reasons,
        persona, req.cards, STATE["movies"],
        natural_language_text=req.natural_language_text,
    )
    return {"reasons": reasons}


@app.post("/type-intro")
def type_intro(req: TypeIntroRequest):
    """사용자 유형 소개 카피 문구 생성(LLM)."""
    persona = {
        "name": req.name,
        "target_type": req.target_type,
        "type_name": _type_name(req.target_type),
        "natural_pref": req.natural_pref,
    }
    return {"type_intro": _llm_call(llm_personalize.generate_type_intro, persona)}
