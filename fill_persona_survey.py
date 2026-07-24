"""
A5 결과를 personas.json에 실제로 채워 넣는다.
meta.empty_fields_note: "survey_answers, scored_axis_scores, predicted_type은 의도적으로 null.
LLM이 프로필만 보고 설문에 응답하게 한 뒤 채워야 검증이 성립함."

- survey_answers: LLM이 프로필만 보고 낸 20문항 응답 {1:5, 2:3, ...}
- scored_axis_scores: mvti.py로 채점한 결과를 target_axis_scores와 같은 형태({I,L,F,P})로 저장
- predicted_type: 채점된 4글자 유형 코드
"""

import json
import os

import mvti
from recommender import load_movies
from validate_survey import simulate_survey_answers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    personas_path = os.path.join(BASE_DIR, "personas.json")
    with open(personas_path, encoding="utf-8") as f:
        personas_data = json.load(f)

    movies = load_movies()
    movies_by_id = {m["movie_id"]: m for m in movies}

    matched = 0
    for i, p in enumerate(personas_data["personas"], 1):
        answers = simulate_survey_answers(p, movies_by_id)
        result = mvti.get_survey_result(answers)

        scored_axis_scores = {
            "I": result["axis_scores"]["세계관"]["I"],
            "L": result["axis_scores"]["무드"]["L"],
            "F": result["axis_scores"]["속도감"]["F"],
            "P": result["axis_scores"]["선택방식"]["P"],
        }

        p["survey_answers"] = answers
        p["scored_axis_scores"] = scored_axis_scores
        p["predicted_type"] = result["type_code"]

        is_match = result["type_code"] == p["target_type"]
        matched += is_match
        print(f"[{i}/{len(personas_data['personas'])}] {p['persona_id']} {p['name']:6s} "
              f"target={p['target_type']} predicted={result['type_code']} {'OK' if is_match else 'MISS'}")

    with open(personas_path, "w", encoding="utf-8") as f:
        json.dump(personas_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\npersonas.json 갱신 완료. 일치율 {matched}/{len(personas_data['personas'])} "
          f"({matched/len(personas_data['personas'])*100:.1f}%)")


if __name__ == "__main__":
    main()
