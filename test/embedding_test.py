"""
임베딩 모델 3종 × 텍스트 구성 4종 비교 평가

페르소나의 natural_pref(자연어 취향)를 쿼리로, 영화 텍스트를 문서로 임베딩해서
코사인 유사도 순위를 매기고 recomm_movies(정답 3편)와 비교
"""

import json
import os
from collections import Counter

import numpy as np

MOVIE_PATH = "/Users/rynn/Desktop/library/프로젝트/캠크루/code/data/movie/movie_db_all.json"
PERSONA_PATH = "/Users/rynn/Desktop/library/프로젝트/캠크루/code/test/personas/personas_final.json"
RESULT_PATH = "/Users/rynn/Desktop/library/프로젝트/캠크루/code/test/embed_results.json"
CACHE_DIR = "/Users/rynn/Desktop/library/프로젝트/캠크루/code/test/embed_cache"

TOP_N = 3
BOOTSTRAP_ROUNDS = 2000
RANDOM_TRIALS = 2000
SEED = 42

MODELS = {
    "KURE-v1": {
        "path": "nlpai-lab/KURE-v1",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "BGE-M3": {
        "path": "BAAI/bge-m3",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "Qwen3-0.6B": {
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": "Instruct: 사용자의 영화 취향 설명에 어울리는 영화를 찾으세요\nQuery:",  # 지시문 기반 모델만 프리픽스 추가
        "doc_prefix": "",
    },
}

VARIANTS = ("overview", "overview+keywords", "description", "description+overview")


# ============================================================
# 데이터
# ============================================================


def load_data():
    with open(MOVIE_PATH, encoding="utf-8") as f:
        movies = json.load(f)
    with open(PERSONA_PATH, encoding="utf-8") as f:
        personas = json.load(f)["personas"]
    return movies, personas


def movie_text(movie: dict, variant: str) -> str:
    overview = movie.get("overview", "")
    description = movie.get("description", "")
    keywords = ", ".join(movie.get("keywords", []))

    if variant == "overview":
        return overview
    if variant == "overview+keywords":
        return f"{overview} {keywords}".strip()
    if variant == "description":
        return description
    if variant == "description+overview":
        return f"{description} {overview}".strip()
    raise ValueError(variant)


def build_truth(movies: list, personas: list):
    """recomm_movies(제목) -> movie_id 인덱스로 변환. 못 찾은 제목은 그대로 보고한다."""
    by_title = {m["korean_title"]: i for i, m in enumerate(movies)}
    normalized = {m["korean_title"].replace(" ", ""): i for i, m in enumerate(movies)}

    truth, unmatched = [], []
    for persona in personas:
        indices = []
        for title in persona.get("recomm_movies", []):
            idx = by_title.get(title, normalized.get(title.replace(" ", "")))
            if idx is None:
                unmatched.append((persona["persona_id"], title))
            else:
                indices.append(idx)
        truth.append(indices)
    return truth, unmatched


# ============================================================
# 임베딩
# ============================================================


def encode(model, texts: list, prefix: str) -> np.ndarray:
    payload = [prefix + t if prefix else t for t in texts]
    vectors = model.encode(payload, batch_size=8, normalize_embeddings=True,
                           show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def cached_encode(model, model_name: str, tag: str, texts: list, prefix: str) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{model_name}__{tag}.npy".replace("/", "_"))
    if os.path.exists(path):
        return np.load(path)
    vectors = encode(model, texts, prefix)
    np.save(path, vectors)
    return vectors


# ============================================================
# 지표
# ============================================================


def rank_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """각 쿼리에 대해 문서를 유사도 내림차순으로 정렬한 인덱스 행렬."""
    similarity = query_vecs @ doc_vecs.T
    return np.argsort(-similarity, axis=1)


def per_persona_metrics(order: np.ndarray, gold: list) -> dict:
    """정답이 몇 위에 있는지 기반으로 개별 지표 계산."""
    position = {doc: rank for rank, doc in enumerate(order, start=1)}
    gold_ranks = sorted(position[g] for g in gold)

    return {
        "hit@3": sum(r <= 3 for r in gold_ranks) / len(gold),
        "recall@5": sum(r <= 5 for r in gold_ranks) / len(gold),
        "recall@10": sum(r <= 10 for r in gold_ranks) / len(gold),
        "mean_rank": float(np.mean(gold_ranks)),
        "mrr": 1.0 / gold_ranks[0],
        "top3": list(order[:TOP_N]),
    }


def aggregate(rows: list) -> dict:
    keys = ("hit@3", "recall@5", "recall@10", "mean_rank", "mrr")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def bootstrap_ci(values: list, rounds: int = BOOTSTRAP_ROUNDS) -> tuple:
    """페르소나를 복원추출해서 평균의 95% 신뢰구간을 구한다."""
    rng = np.random.default_rng(SEED)
    arr = np.asarray(values)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(rounds)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def random_baseline(n_docs: int, truth: list) -> dict:
    """무작위 순위일 때의 기대 성능. 모든 점수는 이 값과 비교해야 의미가 있다."""
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(RANDOM_TRIALS // len(truth) + 1):
        for gold in truth:
            if gold:
                rows.append(per_persona_metrics(rng.permutation(n_docs), gold))
    return aggregate(rows)


def evaluate(order_matrix: np.ndarray, truth: list, personas: list, n_docs: int) -> dict:
    rows, boundary_rows, typical_rows, all_top3 = [], [], [], []

    for order, gold, persona in zip(order_matrix, truth, personas):
        if not gold:
            continue
        metrics = per_persona_metrics(order, gold)
        rows.append(metrics)
        all_top3.extend(metrics["top3"])
        (boundary_rows if persona.get("is_boundary") else typical_rows).append(metrics)

    result = aggregate(rows)
    result["hit@3_ci"] = bootstrap_ci([r["hit@3"] for r in rows])
    result["n"] = len(rows)
    result["coverage"] = len(set(all_top3)) / n_docs
    result["typical_hit@3"] = aggregate(typical_rows)["hit@3"] if typical_rows else None
    result["boundary_hit@3"] = aggregate(boundary_rows)["hit@3"] if boundary_rows else None
    return result


# ============================================================
# 실행
# ============================================================


def main():
    from sentence_transformers import SentenceTransformer

    movies, personas = load_data()
    truth, unmatched = build_truth(movies, personas)

    if unmatched:
        print(f"정답 제목 중 DB에서 못 찾은 항목 {len(unmatched)}개:")
        for pid, title in unmatched:
            print(f"  {pid}: {title}")
        print("  -> 제목 표기를 맞춰야 정확한 평가가 됩니다.\n")

    queries = [p["natural_pref"] for p in personas]
    baseline = random_baseline(len(movies), truth)

    print(f"영화 {len(movies)}편 / 페르소나 {len(personas)}명 "
          f"/ 정답 {sum(len(t) for t in truth)}건")
    print(f"무작위 기준선: hit@3 {baseline['hit@3']:.3f} | "
          f"평균순위 {baseline['mean_rank']:.1f} | MRR {baseline['mrr']:.3f}\n")

    results = {}
    for model_name, config in MODELS.items():
        print(f"[{model_name}] 로딩 중...")
        model = SentenceTransformer(config["path"], trust_remote_code=True)

        query_vecs = cached_encode(model, model_name, "query", queries, config["query_prefix"])

        for variant in VARIANTS:
            texts = [movie_text(m, variant) for m in movies]
            doc_vecs = cached_encode(model, model_name, variant, texts, config["doc_prefix"])
            order = rank_matrix(query_vecs, doc_vecs)
            results[f"{model_name} | {variant}"] = evaluate(order, truth, personas, len(movies))
            print(f"  {variant} 완료")

        del model

    print_report(results, baseline)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {RESULT_PATH}")


def print_report(results: dict, baseline: dict) -> None:
    print("\n" + "=" * 96)
    print(f"{'조합':<28} {'hit@3':>7} {'95% CI':>16} {'R@5':>6} {'R@10':>6} "
          f"{'평균순위':>8} {'MRR':>6} {'커버리지':>8}")
    print("=" * 96)

    for name, r in sorted(results.items(), key=lambda x: -x[1]["hit@3"]):
        low, high = r["hit@3_ci"]
        print(f"{name:<28} {r['hit@3']:>7.3f} {f'[{low:.3f}, {high:.3f}]':>16} "
              f"{r['recall@5']:>6.3f} {r['recall@10']:>6.3f} "
              f"{r['mean_rank']:>8.1f} {r['mrr']:>6.3f} {r['coverage']:>8.2f}")

    print("-" * 96)
    print(f"{'무작위 기준선':<28} {baseline['hit@3']:>7.3f} {'':>16} "
          f"{baseline['recall@5']:>6.3f} {baseline['recall@10']:>6.3f} "
          f"{baseline['mean_rank']:>8.1f} {baseline['mrr']:>6.3f}")

    best = max(results.items(), key=lambda x: x[1]["hit@3"])
    best_low = best[1]["hit@3_ci"][0]
    tied = [n for n, r in results.items() if r["hit@3_ci"][1] >= best_low and n != best[0]]

    print(f"\n1위: {best[0]}")
    if tied:
        print(f"신뢰구간이 겹쳐 통계적으로 구분 안 되는 조합 {len(tied)}개:")
        for name in tied:
            print(f"  {name}")
        print("이 중에서는 속도·모델 크기 같은 실용 기준으로 고르는 게 맞습니다.")
    else:
        print("다른 조합과 신뢰구간이 겹치지 않습니다.")

    print("\n전형/경계 분리:")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["hit@3"]):
        print(f"  {name:<28} 전형 {r['typical_hit@3']:.3f} / 경계 {r['boundary_hit@3']:.3f}")


if __name__ == "__main__":
    main()