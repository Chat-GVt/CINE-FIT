"""
A6: 유형별 영화 분포 산점도용 데이터 준비.
- 50편의 중심화된 4축 벡터를 PCA로 2D 투영 (numpy만 사용, 직접 구현)
- 4축만으로 계산한 쌍별 코사인 유사도 분포 vs 임베딩 쌍별 유사도 분포 비교
  -> "4축만 쓰면 몰리는지" 정량적으로 확인, 임베딩 도입 근거 데이터로 사용
- 결과를 axis_distribution.json으로 저장 (아티팩트에서 그대로 임베드해서 사용)
"""

import json
import math
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AXIS_KEYS = ("I", "L", "F", "P")


def centerize_vector(axis_scores_0_100):
    return {k: (axis_scores_0_100[k] - 50) / 50 for k in AXIS_KEYS}


def pca_2d(matrix: np.ndarray):
    """matrix: (N, D) -> (N, 2) PCA 투영. 표준 라이브러리 대신 numpy 고유값분해로 직접 구현."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    top2 = eigvecs[:, order[:2]]
    explained = eigvals[order[:2]] / eigvals.sum()
    return centered @ top2, explained


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pairwise_sim_stats(vectors: list):
    sims = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(vectors[i], vectors[j]))
    sims_arr = np.array(sims)
    hist, edges = np.histogram(sims_arr, bins=20, range=(-1, 1))
    return {
        "mean": float(sims_arr.mean()),
        "std": float(sims_arr.std()),
        "min": float(sims_arr.min()),
        "max": float(sims_arr.max()),
        "pct_above_0.8": float((sims_arr > 0.8).mean() * 100),
        "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
    }


def dominant_group(axis_vec: dict) -> str:
    """4축 중심화 벡터 -> IL/ID/RL/RD 4개 그룹(세계관+무드 부호)으로 단순 분류(색상용)."""
    i_letter = "I" if axis_vec["I"] >= 0 else "R"
    l_letter = "L" if axis_vec["L"] >= 0 else "D"
    return i_letter + l_letter


def main():
    with open(os.path.join(BASE_DIR, "movie_db_scored.json"), encoding="utf-8") as f:
        movies = json.load(f)
    with open(os.path.join(BASE_DIR, "movie_embeddings.json"), encoding="utf-8") as f:
        embeddings_raw = json.load(f)
    embeddings = {int(k): v for k, v in embeddings_raw.items()}

    axis_vectors = [centerize_vector(m["mvti"]["axis_scores"]) for m in movies]
    axis_matrix = np.array([[v[k] for k in AXIS_KEYS] for v in axis_vectors])

    coords_2d, explained_variance = pca_2d(axis_matrix)

    axis_vec_lists = [[v[k] for k in AXIS_KEYS] for v in axis_vectors]
    embed_vec_lists = [embeddings[m["movie_id"]] for m in movies]

    # 각 영화마다 "4축만으로 볼 때 유사도 0.8 넘는 다른 영화 수" = 몰림 정도
    n = len(movies)
    crowd_counts = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and cosine_similarity(axis_vec_lists[i], axis_vec_lists[j]) > 0.8:
                crowd_counts[i] += 1

    points = []
    for m, vec, (x, y), crowd in zip(movies, axis_vectors, coords_2d, crowd_counts):
        points.append({
            "movie_id": m["movie_id"],
            "title": m["korean_title"],
            "genre": m["genre"],
            "group": dominant_group(vec),
            "axis": vec,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "crowd_count": crowd,
        })

    result = {
        "points": points,
        "explained_variance": [round(float(e) * 100, 1) for e in explained_variance],
        "axis_only_similarity": pairwise_sim_stats(axis_vec_lists),
        "embedding_similarity": pairwise_sim_stats(embed_vec_lists),
    }

    out_path = os.path.join(BASE_DIR, "axis_distribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"PCA 설명 분산: PC1={result['explained_variance'][0]}% PC2={result['explained_variance'][1]}%")
    print(f"\n4축 단독 쌍별 유사도: 평균={result['axis_only_similarity']['mean']:.3f} "
          f"표준편차={result['axis_only_similarity']['std']:.3f} "
          f"0.8 초과 비율={result['axis_only_similarity']['pct_above_0.8']:.1f}%")
    print(f"임베딩 쌍별 유사도:   평균={result['embedding_similarity']['mean']:.3f} "
          f"표준편차={result['embedding_similarity']['std']:.3f} "
          f"0.8 초과 비율={result['embedding_similarity']['pct_above_0.8']:.1f}%")
    print(f"\n{out_path} 저장 완료")


if __name__ == "__main__":
    main()
