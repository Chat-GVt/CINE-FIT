"""
A5: 페르소나 검증 테스트
1. LLM이 페르소나 프로필만 보고 20문항 설문에 응답
2. mvti.py로 채점 -> predicted_type / scored_axis_scores
3. target_type / target_axis_scores와 비교
4. 48명 전체 유형 일치율 산출
"""

import json
import os

import anthropic
from dotenv import load_dotenv

import mvti
from mvti_questions import QUESTIONS
from recommender import load_personas

load_dotenv()
MODEL = "claude-haiku-4-5-20251001"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


SURVEY_SYSTEM_PROMPT = """당신은 주어진 인물 프로필에 완전히 몰입해서, 그 사람이라면
영화 취향 설문 20문항에 실제로 어떻게 응답했을지 추정하는 역할입니다.

각 문항에 대해 1~5점 중 하나로 답하세요.
5=매우 그렇다, 4=조금 그렇다, 3=보통이다, 2=조금 아니다, 1=전혀 아니다

프로필에 나온 나이, 직업, 관람 빈도, 선호 상영관, 자기 취향 설명을 참고하되,
**"좋아요한 영화" 목록의 실제 장르·톤(아래에 함께 제공됨)이 이 사람의 실제 취향을 가장 강하게 보여주는 근거**입니다.
자기 취향 설명 문장 하나만으로 판단하지 말고, 좋아요한 영화들이 공통적으로 어떤 세계관/무드/속도감/대중성을
띠는지 먼저 종합한 뒤, 그 경향에 맞춰 20문항에 답하세요.

**각 축은 괄호 안 설명 문장이 아니라 앞의 숫자(0~100)를 기준으로 판단하세요.**
설명 문장은 뉘앙스 참고용이고, 실제 취향 강도는 숫자가 정답입니다. 특히 여러 편의 숫자를
평균 내어 "이 사람의 좋아요 영화들은 평균적으로 대중성이 몇 점대다"처럼 정량적으로 판단하세요.

프로필에 명시되지 않은 부분은 프로필 전체의 인상에서 자연스럽게 추론하세요."""


def _movie_signal_line(movie: dict) -> str:
    axis = movie["mvti"]["axis_scores"]
    reasons = movie["mvti"]["reasons"]
    return (f"- {movie['korean_title']} ({', '.join(movie['genre'])}) : "
            f"세계관 {axis['I']}/100({reasons['I']}) / "
            f"무드 {axis['L']}/100({reasons['L']}) / "
            f"속도감 {axis['F']}/100({reasons['F']}) / "
            f"대중성 {axis['P']}/100({reasons['P']})")


def build_survey_user_prompt(persona: dict, movies_by_id: dict) -> str:
    liked_movies = [movies_by_id[i] for i in persona["liked_movie_ids"] if i in movies_by_id]
    watched_titles = [movies_by_id[i]["korean_title"] for i in persona["watched_movie_ids"] if i in movies_by_id]
    liked_block = "\n".join(_movie_signal_line(m) for m in liked_movies) if liked_movies else "없음"

    questions_block = "\n".join(f"{n}. {text}" for n, text in sorted(QUESTIONS.items()))

    return f"""[인물 프로필]
이름: {persona['name']} ({persona['age']}세, {persona['gender']})
직업: {persona['occupation']}
관람 빈도: {persona['watch_freq']}
선호 상영관: {persona['preferred_theater']}
자기 취향 설명: "{persona['natural_pref']}"
좋아요한 영화(장르·실제 취향축 포함):
{liked_block}
시청한 영화: {', '.join(watched_titles) if watched_titles else '없음'}

[설문 20문항]
{questions_block}

위 프로필의 인물이라면 각 문항에 1~5점으로 어떻게 답할지 20개 전부 응답하세요."""


def _build_answer_schema() -> dict:
    return {
        "type": "object",
        "properties": {str(n): {"type": "integer"} for n in range(1, 21)},
        "required": [str(n) for n in range(1, 21)],
        "additionalProperties": False,
    }


def simulate_survey_answers(persona: dict, movies_by_id: dict) -> dict:
    """페르소나 프로필을 보고 LLM이 20문항에 응답 -> {1: 5, 2: 3, ...}"""
    user_prompt = build_survey_user_prompt(persona, movies_by_id)
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=500,
        system=SURVEY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": _build_answer_schema()}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    raw = json.loads(text)
    return {int(k): v for k, v in raw.items()}


def main():
    from recommender import load_movies

    personas = load_personas()
    movies = load_movies()
    movies_by_id = {m["movie_id"]: m for m in movies}

    matched = 0
    axis_diffs = {"I": [], "L": [], "F": [], "P": []}

    for p in personas:
        answers = simulate_survey_answers(p, movies_by_id)
        result = mvti.get_survey_result(answers)
        predicted_type = result["type_code"]
        target_type = p["target_type"]
        is_match = predicted_type == target_type
        matched += is_match

        scored = {
            "I": result["axis_scores"]["세계관"]["I"],
            "L": result["axis_scores"]["무드"]["L"],
            "F": result["axis_scores"]["속도감"]["F"],
            "P": result["axis_scores"]["선택방식"]["P"],
        }
        for k in axis_diffs:
            axis_diffs[k].append(abs(scored[k] - p["target_axis_scores"][k]))

        tag = "OK " if is_match else "MISS"
        print(f"[{tag}] {p['persona_id']} {p['name']:6s} target={target_type} predicted={predicted_type} "
              f"| scored={scored} target_axis={p['target_axis_scores']}")

    print(f"\n=== 유형 일치율: {matched}/{len(personas)} ({matched/len(personas)*100:.1f}%) ===")
    for k, diffs in axis_diffs.items():
        avg = sum(diffs) / len(diffs)
        print(f"  {k}축 평균 오차: {avg:.1f}점")


if __name__ == "__main__":
    main()
