"""
B6: LLM(Anthropic Claude Haiku 4.5) 기반 자연어 -> MVTI 4축(I/L/F/P) 변환
"""

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = "claude-haiku-4-5-20251001"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


AXIS_SYSTEM_PROMPT = """당신은 CINE:FIT의 영화 취향 분석가입니다.
사용자가 자유롭게 쓴 영화 취향 설명을 읽고, MVTI 4축 점수를 0~100 사이 정수로 추정하세요.

- I(세계관): 0=완전히 현실적(Realism) ~ 100=완전히 상상적(Imagination)
- L(무드): 0=완전히 어두움(Dark) ~ 100=완전히 밝음(Light)
- F(속도감): 0=매우 느림(Slow) ~ 100=매우 빠름(Fast)
- P(선택방식): 0=독창적·마이너(Unique) ~ 100=대중적·잘 알려진(Popular)

규칙:
- 텍스트에서 명확히 드러나는 축만 50에서 크게 벗어나게 추정하고, 언급이 없거나 애매한 축은 50(중립)에 가깝게 두세요.
- 과도하게 극단적인 값(0이나 100)은 텍스트가 그 정도로 명확할 때만 쓰세요.
- 각 축마다 왜 그렇게 추정했는지 한 줄 근거도 함께 반환하세요."""

AXIS_SCHEMA = {
    "type": "object",
    "properties": {
        "I": {"type": "integer"},
        "L": {"type": "integer"},
        "F": {"type": "integer"},
        "P": {"type": "integer"},
        "reasons": {
            "type": "object",
            "properties": {
                "I": {"type": "string"},
                "L": {"type": "string"},
                "F": {"type": "string"},
                "P": {"type": "string"},
            },
            "required": ["I", "L", "F", "P"],
            "additionalProperties": False,
        },
    },
    "required": ["I", "L", "F", "P", "reasons"],
    "additionalProperties": False,
}


def convert_text_to_axis_scores(natural_language_text: str) -> dict:
    """자연어 취향 설명 -> {"I":.., "L":.., "F":.., "P":.., "reasons": {...}}"""
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=400,
        system=AXIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"취향 설명: {natural_language_text}"}],
        output_config={"format": {"type": "json_schema", "schema": AXIS_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


if __name__ == "__main__":
    samples = [
        "혼자 조용히 볼 수 있는 잔잔한 영화가 좋아요",
        "무겁고 여운 남는 거요. 실화 기반이면 더 좋고",
        "웃기고 시원한 한국 상업영화요. 팝콘 먹으면서 아무 생각 없이 보기 딱이죠",
        "봉준호 감독 영화처럼 현실적이면서도 상상 밖의 전개를 좋아해요",
        "좀비가 나오는 무서운 영화 보고 싶어요",
        "때리고 부수는 한국 범죄 액션이요. 답답한 거 없이 사이다처럼 가는 게 좋아요",
    ]
    for text in samples:
        result = convert_text_to_axis_scores(text)
        print(f"\n입력: {text}")
        print(f"  I={result['I']} L={result['L']} F={result['F']} P={result['P']}")
        for axis, reason in result["reasons"].items():
            print(f"    {axis}: {reason}")
