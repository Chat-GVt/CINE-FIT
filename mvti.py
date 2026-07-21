"""
CGV MVTI 설문 채점 모듈
- 20문항 응답(1~5점) -> 4축 성향(%) 계산 -> 16유형 코드 매핑
"""

from typing import Dict, Tuple

# ---------- 축 정의 ----------
# primary_qs: 원점수 그대로 사용하는 문항
# reverse_qs: 역채점(6 - 원점수) 적용하는 문항
AXES = {
    "세계관":   {"primary": "I", "secondary": "R", "primary_qs": [9, 14, 18], "reverse_qs": [4, 7]},
    "무드":     {"primary": "L", "secondary": "D", "primary_qs": [2, 11, 20], "reverse_qs": [6, 15]},
    "속도감":   {"primary": "F", "secondary": "S", "primary_qs": [3, 12, 19], "reverse_qs": [8, 16]},
    "선택방식": {"primary": "P", "secondary": "U", "primary_qs": [5, 13, 17], "reverse_qs": [1, 10]},
}

TOTAL_QUESTIONS = 20

# ---------- 16유형 정보 (유형명 / 키워드 / 대표 캐릭터) ----------
TYPE_INFO = {
    "ILFP": {"name": "화려한 마블 히어로형", "keywords": "상상·밝음·빠름·대중",
             "characters": ["아이언맨 (어벤져스)", "해리 포터 (해리포터 시리즈)", "피터 파커/스파이더맨 (스파이더맨 시리즈)"]},
    "ILFU": {"name": "지하실의 크리에이터형", "keywords": "상상·밝음·빠름·독창",
             "characters": ["에드워드 시저핸즈 (가위손)", "잭 스켈링턴 (크리스마스의 악몽)", "코렐라린 (코렐라인: 비밀의 문)"]},
    "ILSP": {"name": "따뜻한 가족형", "keywords": "상상·밝음·느림·대중",
             "characters": ["기쁨이 (인사이드 아웃)", "러셀 (업)", "미겔 (코코)"]},
    "ILSU": {"name": "기묘한 몽상가형", "keywords": "상상·밝음·느림·독창",
             "characters": ["아멜리 (아멜리에)", "오필리아 (판의 미로)", "월-E (월-E)"]},
    "IDFP": {"name": "어둠의 히어로형", "keywords": "상상·어두움·빠름·대중",
             "characters": ["배트맨 (다크나이트)", "다스 베이더 (스타워즈)", "네오 (매트릭스)"]},
    "IDFU": {"name": "고독한 파이터형", "keywords": "상상·어두움·빠름·독창",
             "characters": ["알렉스 (시계태엽 오렌지)", "타일러 더든 (파이트 클럽)", "퓨리오사 (매드맥스: 분노의 도로)"]},
    "IDSP": {"name": "느긋한 우주여행가형", "keywords": "상상·어두움·느림·대중",
             "characters": ["쿠퍼 (인터스텔라)", "라이언 스톤 (그래비티)", "릭 데커드 (블레이드 러너)"]},
    "IDSU": {"name": "심연의 수집가형", "keywords": "상상·어두움·느림·독창",
             "characters": ["로이 배티 (블레이드 러너)", "크리스 켈빈 (솔라리스)", "헨리 스펜서 (이레이저 헤드)"]},
    "RLFP": {"name": "유쾌한 이야기꾼형", "keywords": "현실·밝음·빠름·대중",
             "characters": ["고반장 (극한직업)", "포레스트 검프 (포레스트 검프)", "로키 발보아 (록키)"]},
    "RLFU": {"name": "골목의 큐레이터형", "keywords": "현실·밝음·빠름·독창",
             "characters": ["주노 맥거프 (주노)", "월터 미티 (월터의 상상은 현실이 된다)", "사치에 (카모메 식당)"]},
    "RLSP": {"name": "힐링의 관람자형", "keywords": "현실·밝음·느림·대중",
             "characters": ["덕수 (국제시장)", "셀린 & 제시 (비포 선라이즈)", "와타나베 히로코 (러브레터)"]},
    "RLSU": {"name": "꼼꼼한 평론가형", "keywords": "현실·밝음·느림·독창",
             "characters": ["패터슨 (패터슨)", "이치코 (리틀 포레스트)", "오사무 (어느 가족)"]},
    "RDFP": {"name": "냉정한 추적자형", "keywords": "현실·어두움·빠름·대중",
             "characters": ["마석도 (범죄도시)", "고니 (타짜)", "마이클 코를레오네 (대부)"]},
    "RDFU": {"name": "뒷골목의 비판자형", "keywords": "현실·어두움·빠름·독창",
             "characters": ["오대수 (올드보이)", "트래비스 비클 (택시 드라이버)", "이자성 (신세계)"]},
    "RDSP": {"name": "역사 속 주인공형", "keywords": "현실·어두움·느림·대중",
             "characters": ["만섭 (택시운전사)", "오펜하이머 (오펜하이머)", "박두만 (살인의 추억)"]},
    "RDSU": {"name": "독립영화 감독형", "keywords": "현실·어두움·느림·독창",
             "characters": ["신애 (밀양)", "샤이론 (문라이트)", "안톤 시거 (노인을 위한 나라는 없다)"]},
}


def _validate_answers(answers: Dict[int, int]) -> None:
    missing = [q for q in range(1, TOTAL_QUESTIONS + 1) if q not in answers]
    if missing:
        raise ValueError(f"응답이 누락된 문항: {missing}")
    invalid = [(q, v) for q, v in answers.items() if not (1 <= v <= 5)]
    if invalid:
        raise ValueError(f"1~5 범위를 벗어난 응답: {invalid}")


def _score_axis(answers: Dict[int, int], primary_qs, reverse_qs) -> float:
    raw_scores = [answers[q] for q in primary_qs]
    reverse_scores = [6 - answers[q] for q in reverse_qs]
    avg = sum(raw_scores + reverse_scores) / len(primary_qs + reverse_qs)
    primary_pct = (avg - 1) / 4 * 100  # 1~5점 척도를 0~100%로 정규화
    return round(primary_pct, 1)


def calculate_mvti(answers: Dict[int, int]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    answers: {문항번호(1~20): 응답값(1~5)}
    return: (4글자 유형 코드, 축별 상세 점수)
    """
    _validate_answers(answers)

    axis_scores: Dict[str, Dict[str, float]] = {}
    type_code = ""

    for axis_name, cfg in AXES.items():
        primary_pct = _score_axis(answers, cfg["primary_qs"], cfg["reverse_qs"])
        secondary_pct = round(100 - primary_pct, 1)

        # 50% 기준으로 우세한 방향을 유형 코드 글자로 채택 (정확히 50이면 primary 채택)
        if primary_pct >= 50:
            dominant_label, dominant_pct = cfg["primary"], primary_pct
        else:
            dominant_label, dominant_pct = cfg["secondary"], secondary_pct

        axis_scores[axis_name] = {
            cfg["primary"]: primary_pct,
            cfg["secondary"]: secondary_pct,
            "dominant": dominant_label,
            "dominant_pct": dominant_pct,
        }
        type_code += dominant_label

    return type_code, axis_scores


def get_survey_result(answers: Dict[int, int]) -> Dict:
    """설문 응답 -> 유형 코드 + 축 점수 + 유형 정보(이름/키워드/캐릭터)까지 한번에 반환"""
    type_code, axis_scores = calculate_mvti(answers)
    type_info = TYPE_INFO.get(type_code, {})
    return {
        "type_code": type_code,  # MVTI 유형
        "axis_scores": axis_scores,  # 축별 퍼센트
        "type_name": type_info.get("name"),  # MVTI 유형 이름
        "keywords": type_info.get("keywords"),  # MVTI 유형 키워드
        "characters": type_info.get("characters"),  # 관련 캐릭터
    }


if __name__ == "__main__":
    # 예시: 실제로는 앱에서 문항 1~20에 대한 유저 응답을 이 형태로 수집
    sample_answers = {
        1: 2, 2: 4, 3: 3, 4: 2, 5: 2,
        6: 4, 7: 2, 8: 2, 9: 5, 10: 4,
        11: 4, 12: 4, 13: 2, 14: 5, 15: 4,
        16: 2, 17: 2, 18: 5, 19: 3, 20: 4,
    }

    result = get_survey_result(sample_answers)

    print(f"유형 코드: {result['type_code']}")
    print(f"유형명: {result['type_name']} ({result['keywords']})")
    print(f"대표 캐릭터: {result['characters']}")
    print("축별 상세 점수:")
    for axis, scores in result["axis_scores"].items():
        print(f"  {axis}: {scores}")