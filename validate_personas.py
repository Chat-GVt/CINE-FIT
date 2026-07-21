"""
B4 검증: personas.json 48명 전원에게 recommend_for_persona()를 돌려
유형별로 추천 결과가 실제로 달라지는지 확인한다.
"""

from collections import Counter

from recommender import load_movies, load_movie_embeddings, load_personas, recommend_for_persona


def main():
    movies = load_movies()
    movie_embeddings = load_movie_embeddings()
    personas = load_personas()

    top1_counter = Counter()
    top1_by_type = {}

    for p in personas:
        cards = recommend_for_persona(p, movies, movie_embeddings)
        top1 = cards[0] if cards else None
        top1_title = top1["title"] if top1 else "추천없음"
        top1_counter[top1_title] += 1
        top1_by_type.setdefault(p["target_type"], []).append(top1_title)

        tag = f"(boundary:{p['boundary_axis']})" if p["is_boundary"] else ""
        print(f"{p['persona_id']} {p['name']:6s} {p['target_type']} {tag:14s} "
              f"-> [{top1['similarity']}%] {top1_title}")

    print("\n=== 유형별 top1 (3명씩) ===")
    for t, titles in sorted(top1_by_type.items()):
        flag = "" if len(set(titles)) > 1 else "  <- 3명 다 동일"
        print(f"  {t}: {titles}{flag}")

    print(f"\n서로 다른 top1 영화 수: {len(top1_counter)} / 48명")
    print("가장 많이 겹친 영화 top5:", top1_counter.most_common(5))


if __name__ == "__main__":
    main()
