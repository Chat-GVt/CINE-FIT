"""
B6 검증: personas.json 각자의 natural_pref(자유서술 취향, 실제 온보딩 UI의 그 텍스트)를
자연어 입력으로 사용했을 때 vs 안 했을 때 추천이 어떻게 달라지는지 비교한다.
"""

from collections import Counter

from recommender import (
    load_movies, load_movie_embeddings, load_personas, load_snacks, load_goods,
    recommend_for_persona,
)


def main():
    movies = load_movies()
    movie_embeddings = load_movie_embeddings()
    personas = load_personas()
    snacks = load_snacks()
    goods = load_goods()

    # 몇 명은 자세히(before/after 비교)
    sample_ids = ["P-001", "P-013", "P-025", "P-037"]
    for pid in sample_ids:
        p = next(pp for pp in personas if pp["persona_id"] == pid)
        print(f"\n=== {p['persona_id']} {p['name']} ({p['target_type']}) ===")
        print(f"  natural_pref: {p['natural_pref']}")

        without = recommend_for_persona(p, movies, movie_embeddings, snacks, goods)
        with_nl = recommend_for_persona(p, movies, movie_embeddings, snacks, goods,
                                          natural_language_text=p["natural_pref"])

        print("  [자연어 없이(좋아요 이력 기반)]")
        for c in without:
            print(f"    [{c['similarity']}%] {c['title']}")
        print("  [natural_pref 반영]")
        for c in with_nl:
            print(f"    [{c['similarity']}%] {c['title']}")

    # 48명 전체 top1 변화 요약
    changed = 0
    top1_counter_with_nl = Counter()
    for p in personas:
        without = recommend_for_persona(p, movies, movie_embeddings, snacks, goods)
        with_nl = recommend_for_persona(p, movies, movie_embeddings, snacks, goods,
                                          natural_language_text=p["natural_pref"])
        top1_counter_with_nl[with_nl[0]["title"]] += 1
        if without[0]["title"] != with_nl[0]["title"]:
            changed += 1

    print(f"\n=== 48명 전체 요약 ===")
    print(f"natural_pref 반영으로 top1이 바뀐 페르소나 수: {changed} / 48")
    print(f"natural_pref 반영 시 서로 다른 top1 영화 수: {len(top1_counter_with_nl)} / 48")


if __name__ == "__main__":
    main()
