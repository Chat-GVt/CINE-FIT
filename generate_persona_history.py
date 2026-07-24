"""
A4: personas.json의 recent_movies(자유 텍스트, 우리 movie_db와 거의 안 겹침)를 지우고
movie_id 기반 좋아요/시청 이력을 새로 생성한다.

방식:
- persona.target_axis_scores를 중심화(-1~+1)한 벡터 vs 각 영화 mvti.axis_scores(중심화)
  코사인 유사도로 랭킹
- 개봉 전(upcoming) 영화는 "시청"할 수 없으므로 후보에서 제외
- watch_freq에 따라 시청 편수를 다르게 부여 (자주 볼수록 이력 많음)
- 시청한 영화 중 유사도 상위 절반을 "좋아요"로 표시 (좋아요 ⊆ 시청)
"""

import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WATCH_COUNT_BY_FREQ = {
    "주 1회 이상": 8,
    "월 2~3회": 5,
    "월 1회": 3,
}


def centerize_vector(axis_scores_0_100):
    return {k: (axis_scores_0_100[k] - 50) / 50 for k in ("I", "L", "F", "P")}


AXIS_WEIGHTS = {"I": 1.0, "L": 1.0, "F": 1.0, "P": 2.0}


def cosine_similarity(vec_a, vec_b, weights=AXIS_WEIGHTS):
    """
    가중 코사인 유사도. P축 가중치를 높여서(기본 2배), 좋아요 영화 선정 시
    I/L/F만 맞고 P는 희생되는 경우를 줄인다(A5 검증에서 P축 오차가 컸던 원인).
    """
    keys = [k for k in vec_a if k in vec_b]
    dot = sum(weights[k] * vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(weights[k] * vec_a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(weights[k] * vec_b[k] ** 2 for k in keys))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    personas_path = os.path.join(BASE_DIR, "personas.json")
    movies_path = os.path.join(BASE_DIR, "movie_db_scored.json")

    with open(personas_path, encoding="utf-8") as f:
        personas_data = json.load(f)
    with open(movies_path, encoding="utf-8") as f:
        movies = json.load(f)

    eligible_movies = [m for m in movies if m["status"] != "upcoming"]
    print(f"시청 가능 영화(개봉전 제외): {len(eligible_movies)} / {len(movies)}")

    for p in personas_data["personas"]:
        user_vector = centerize_vector(p["target_axis_scores"])

        ranked = sorted(
            eligible_movies,
            key=lambda m: cosine_similarity(user_vector, centerize_vector(m["mvti"]["axis_scores"])),
            reverse=True,
        )

        watch_count = WATCH_COUNT_BY_FREQ[p["watch_freq"]]
        watched = ranked[:watch_count]
        like_count = max(1, (watch_count + 1) // 2)
        liked = watched[:like_count]

        p.pop("recent_movies", None)
        p["watched_movie_ids"] = [m["movie_id"] for m in watched]
        p["liked_movie_ids"] = [m["movie_id"] for m in liked]

    with open(personas_path, "w", encoding="utf-8") as f:
        json.dump(personas_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("personas.json 갱신 완료")

    # 검증 출력
    for p in personas_data["personas"][:3]:
        print(f"\n{p['persona_id']} {p['name']} ({p['target_type']}, watch_freq={p['watch_freq']})")
        print(f"  watched_movie_ids: {p['watched_movie_ids']}")
        print(f"  liked_movie_ids:   {p['liked_movie_ids']}")
        id_to_title = {m["movie_id"]: m["korean_title"] for m in movies}
        print(f"  watched titles: {[id_to_title[i] for i in p['watched_movie_ids']]}")
        print(f"  liked titles:   {[id_to_title[i] for i in p['liked_movie_ids']]}")


if __name__ == "__main__":
    main()
