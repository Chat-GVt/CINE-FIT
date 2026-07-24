"""
B5: LLM(Anthropic Claude Haiku 4.5) 기반 개인화 문장 생성
- generate_type_intro: 유형 소개글
- generate_curation_reason: 추천 3편을 "묶음 단위"로 설명하는 큐레이션 이유
"""

import json
import os

import anthropic
from dotenv import load_dotenv

import mvti

load_dotenv()
MODEL = "claude-haiku-4-5-20251001"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def generate_type_intro(persona: dict) -> str:
    prompt = f"""당신은 CGV 영화 취향 분석 서비스 'CINE:FIT'의 카피라이터입니다.
아래 사용자 정보를 보고, 유형을 소개하는 문구를 한국어로 작성하세요.

조건:
- 1~2문장, 40~70자 내외
- 사용자의 "취향 메모"에 나온 구체적인 표현(단어, 뉘앙스)을 하나 이상 자연스럽게 녹여낼 것
- 문장 구조를 매번 다르게 — 아래 예시들의 어순/구조를 그대로 반복하지 말 것
- 다정하거나 위트있는 톤. 유형 코드와 유형명은 자연스럽게 언급

예시(참고용 — 이 구조를 복사하지 말고 매번 다르게 쓸 것):
- "액션이면 무조건 큰 화면부터 찾으신다는 도현님, 딱 화려한 마블 히어로형(ILFP)이시네요."
- "잔잔하고 조용한 영화를 좋아하신다니 — 세라님은 힐링의 관람자형(RLSP), 마음의 속도가 남다르시군요."

이름: {persona['name']}
유형 코드: {persona['target_type']}
유형명: {persona['type_name']}
취향 메모: {persona['natural_pref']}

출력은 소개 문구만, 다른 설명 없이."""

    resp = get_client().messages.create(
        model=MODEL, max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


CURATION_SYSTEM_PROMPT = """너는 CINE:FIT의 큐레이터다. 사용자에게 추천된 영화 3편을 왜 이렇게 골랐는지
"묶음 단위"로 설명한다. 개별 영화 소개가 아니라 큐레이션의 관점을 쓴다.

[작성 방식]
1. 3편이 공유하는 결을 한 문장으로 먼저 짚는다. (가장 중요)
2. 그 다음 한 문장으로 보충한다. 예를 들면:
   - 3편의 미묘한 차이와 "무엇부터 볼지" 가이드
   - 관람 시 유의점 (러닝타임, 호흡, 분위기)
   - 사용자가 입력한 취향과의 연결
3. 총 2문장, 100자 이내.

[반드시 지킬 것]
- 영화 제목을 3개 다 나열하지 마라. 이미 카드에 보인다.
  필요하면 "첫 번째", "세 번째"로 지칭한다.
- 줄거리를 지어내지 마라. 제공된 분석 문장 안에서만 쓴다.
- 스포일러 금지. 결말이나 반전을 암시하지 마라.
- 유형 코드(RDFP 등)를 노출하지 마라.
- 사용자가 자연어 취향을 입력했다면, 그 표현을 되받아서 연결한다.
- 이모지 금지. 존댓말."""

CURATION_SCHEMA = {
    "type": "object",
    "properties": {"curation_reason": {"type": "string"}},
    "required": ["curation_reason"],
    "additionalProperties": False,
}


def generate_curation_reason(
    persona: dict,
    cards: list,
    movies: list,
    natural_language_text: str = None,
) -> str:
    """추천된 top3 카드를 묶음으로 왜 골랐는지 큐레이션 관점의 문장을 생성."""
    movies_by_title = {m["korean_title"]: m for m in movies}
    type_info = mvti.TYPE_INFO.get(persona["target_type"], {})
    axis = persona["target_axis_scores"]

    movie_lines = []
    for i, c in enumerate(cards, 1):
        m = movies_by_title[c["title"]]
        reasons = m["mvti"]["reasons"]
        movie_lines.append(
            f"{i}위 {c['title']} ({c['similarity']}%) — {', '.join(c['genre'])}, {m['runtime']}분\n"
            f"    특성: I={reasons['I']} / L={reasons['L']} / F={reasons['F']} / P={reasons['P']}"
        )

    user_prompt = f"""[사용자]
유형: {persona['type_name']} ({type_info.get('keywords', '')})
취향: 세계관 {axis['I']}/100, 무드 {axis['L']}/100, 속도감 {axis['F']}/100, 선택 {axis['P']}/100
직접 입력: {natural_language_text or '없음'}

[추천된 3편]
{chr(10).join(movie_lines)}

이 3편을 왜 골랐는지 큐레이션 관점에서 작성해라."""

    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        system=CURATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": CURATION_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["curation_reason"]


if __name__ == "__main__":
    import recommender as r

    movies = r.load_movies()
    movie_embeddings = r.load_movie_embeddings()
    personas = r.load_personas()
    snacks = r.load_snacks()
    goods = r.load_goods()

    for pid in ["P-037", "P-001", "P-023", "P-031"]:
        p = next(pp for pp in personas if pp["persona_id"] == pid)
        cards = r.recommend_for_persona(p, movies, movie_embeddings, snacks, goods, top_n=3)

        print(f"\n=== {p['persona_id']} {p['name']} ({p['target_type']} · {p['type_name']}) ===")
        for c in cards:
            print(f"  추천: {c['title']} ({c['similarity']}%)")
        print("큐레이션 이유:", generate_curation_reason(p, cards, movies))

    # 자연어 입력이 있는 케이스
    p46 = next(pp for pp in personas if pp["persona_id"] == "P-046")
    nl_text = "무겁고 여운 남는 거요. 실화 기반이면 더 좋고"
    cards46 = r.recommend_for_persona(p46, movies, movie_embeddings, snacks, goods, top_n=3,
                                        natural_language_text=nl_text)
    print(f"\n=== {p46['persona_id']} {p46['name']} ({p46['target_type']} · {p46['type_name']}) + 자연어: \"{nl_text}\" ===")
    for c in cards46:
        print(f"  추천: {c['title']} ({c['similarity']}%)")
    print("큐레이션 이유:", generate_curation_reason(p46, cards46, movies, natural_language_text=nl_text))
