"""
16개 유형(경계형 제외, 대표 1명씩)의 추천 결과를 recommendation_results.md 파일로 저장.
매번 스크립트 다시 안 돌려도 파일 열어서 바로 확인 가능.
"""

import os

import recommender as r
import llm_personalize as lp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    movies = r.load_movies()
    movie_embeddings = r.load_movie_embeddings()
    personas = r.load_personas()
    snacks = r.load_snacks()
    goods = r.load_goods()

    seen_types = set()
    selected = []
    for p in personas:
        if p["target_type"] not in seen_types and not p["is_boundary"]:
            seen_types.add(p["target_type"])
            selected.append(p)

    lines = ["# CINE:FIT 페르소나별 추천 결과 (유형당 대표 1명)\n"]
    lines.append(f"총 {len(selected)}개 유형, 각 유형의 대표(경계형 아닌) 페르소나 1명씩\n")
    lines.append("---\n")

    for p in selected:
        cards = r.recommend_for_persona(p, movies, movie_embeddings, snacks, goods, top_n=3)
        reason = lp.generate_curation_reason(p, cards, movies)
        type_intro = lp.generate_type_intro(p)

        lines.append(f"## {p['persona_id']} {p['name']} — {p['target_type']} ({p['type_name']})\n")
        lines.append(f"> {type_intro}\n")
        lines.append(f"- **자기소개**: {p['natural_pref']}")
        lines.append(f"- **target_axis_scores**: {p['target_axis_scores']}\n")

        lines.append("| 순위 | 영화 | 유사도 | 장르 | 특별관 | 매점 | 굿즈 |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, c in enumerate(cards, 1):
            screen = c["screen"]["screen"] if c["screen"] else "없음"
            snack_names = ", ".join(s["name"] for s in c["snacks"]) or "없음"
            goods_names = ", ".join(g["name"] for g in c["goods"]) or "없음"
            lines.append(f"| {i} | {c['title']} | {c['similarity']}% | {', '.join(c['genre'])} | "
                          f"{screen} | {snack_names} | {goods_names} |")

        lines.append(f"\n**추천 이유(LLM 큐레이션)**: {reason}\n")
        lines.append("---\n")

    out_path = os.path.join(BASE_DIR, "recommendation_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
