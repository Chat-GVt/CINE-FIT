"""
CINE:FIT 추천 파이프라인 설정
"""

import os

# ---------- 경로 ----------
# recommend/ 의 부모(프로젝트 루트) 아래 front/ 에 데이터가 있다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # .../CINE-FIT-main/recommend
ROOT_DIR = os.path.dirname(BASE_DIR)                       # .../CINE-FIT-main
FRONT_DIR = os.path.join(ROOT_DIR, "front")

MOVIE_DB_PATH = os.path.join(FRONT_DIR, "movie_db_all.json")
MOVIE_EMBEDDING_PATH = os.path.join(BASE_DIR, "movie_embeddings.json")  # 자동 생성/갱신되는 캐시
QUESTION_PATH = os.path.join(FRONT_DIR, "questions.json")
TYPE_PATH = os.path.join(FRONT_DIR, "types.json")         # 16유형 이름 + 대표 캐릭터
USER_INPUT_PATH = os.path.join(BASE_DIR, "user_input.json")

# ---------- MVTI 4축 ----------
# key = 점수가 높을 때의 극
AXIS_KEYS = ("I", "L", "F", "P")

AXIS_POLES = {
    "I": {"high": ("I", "상상"), "low": ("R", "현실")},
    "L": {"high": ("L", "밝음"), "low": ("D", "어두움")},
    "F": {"high": ("F", "빠름"), "low": ("S", "느림")},
    "P": {"high": ("P", "대중"), "low": ("U", "독창")},
}

# 리커트 척도 범위 (questions.json의 문항이 모두 이 범위)
LIKERT_MIN = 1
LIKERT_MAX = 5

# ---------- 추천 점수 가중치 ----------
# 최종 점수 = W_AXIS * (MVTI 축 유사도) + W_EMBED * (텍스트 임베딩 유사도)
# 둘을 뒤집고 싶으면 이 두 줄만 바꾸면 됨.
W_AXIS = 0.7
W_EMBED = 0.3

TOP_N = 3

# ---------- 모델 ----------
# 한국어 특화 임베딩(KURE-v1). 영화 description ↔ 사용자 자연어 취향 비교에 사용.
EMBEDDING_REPO = "nlpai-lab/KURE-v1"
EMBEDDING_MAX_LENGTH = 1024  # description이 길어 넉넉히

OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
