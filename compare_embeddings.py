"""
임베딩 모델별 페르소나 추천 결과 비교 뷰어.

movie_embeddings.json(BGE-M3)과 movie_embeddings_qwen3.json(Qwen3)을 각각 넣어
같은 페르소나에 대해 top-3 추천을 나란히 출력한다.
recommender.recommend_for_persona() 가 임베딩 dict를 인자로 받으므로 recommender.py는 수정 불필요.

사용:  python compare_embeddings.py [persona_id ...]
       인자 없으면 대표 4명(P-001, P-022, P-031, P-040).
"""

import json
import os
import sys

import recommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("BGE-M3", "movie_embeddings.json"),
    ("Qwen3-0.6B", "movie_embeddings_qwen3.json"),
]


def load_embeddings(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    movies = recommender.load_movies()
    snacks = recommender.load_snacks()
    goods = recommender.load_goods()
    personas = {p["persona_id"]: p for p in recommender.load_personas()}

    embed_sets = []
    for label, fname in MODELS:
        emb = load_embeddings(fname)
        if emb is None:
            print(f"[건너뜀] {label}: {fname} 없음")
            continue
        dim = len(next(iter(emb.values())))
        embed_sets.append((label, emb, dim))

    ids = sys.argv[1:] or ["P-001", "P-022", "P-031", "P-040"]

    for pid in ids:
        p = personas.get(pid)
        if p is None:
            print(f"\n[없는 페르소나] {pid}")
            continue
        print(f"\n{'='*70}")
        print(f"{pid} {p['name']} ({p['target_type']}) | liked={p['liked_movie_ids']} watched={p['watched_movie_ids']}")
        print('='*70)

        for label, emb, dim in embed_sets:
            cards = recommender.recommend_for_persona(p, movies, emb, snacks, goods, top_n=3)
            print(f"\n  [{label}] dim={dim}")
            for rank, c in enumerate(cards, 1):
                print(f"    {rank}. {c['title']:<24} "
                      f"최종 {c['similarity']:>5}%  (축 {c['axis_similarity']}% / 임베딩 {c['embed_similarity']}%)  "
                      f"{c['genre']}")


if __name__ == "__main__":
    main()
