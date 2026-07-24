"""
A6: axis_distribution.json을 matplotlib으로 그려서 PNG 파일로 저장.
(아티팩트 웹뷰어 대신 로컬에서 바로 열어볼 수 있게)
"""

import json
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 한글 폰트 설정 (Windows 기본 맑은 고딕)
for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    with open(os.path.join(BASE_DIR, "axis_distribution.json"), encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- 왼쪽: PCA 산점도 ----
    ax = axes[0]
    xs = [p["x"] for p in data["points"]]
    ys = [p["y"] for p in data["points"]]
    crowd = [p["crowd_count"] for p in data["points"]]
    titles = [p["title"] for p in data["points"]]

    sc = ax.scatter(xs, ys, c=crowd, cmap="Blues", s=120, edgecolors="white", linewidths=1, vmin=0)
    for x, y, t in zip(xs, ys, titles):
        pass  # 라벨은 너무 많아서 생략, 대신 몰림 심한 것만 아래에서 표시

    # 몰림 수 상위 8개만 라벨 표시
    top_crowd = sorted(zip(crowd, titles, xs, ys), reverse=True)[:8]
    for c, t, x, y in top_crowd:
        ax.annotate(t[:10], (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title(f"영화 50편 4축 벡터 PCA 투영 (PC1 {data['explained_variance'][0]}%, PC2 {data['explained_variance'][1]}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("몰림 수 (4축 유사도 0.8 초과 다른 영화 개수)")

    # ---- 오른쪽: 유사도 분포 히스토그램 ----
    ax2 = axes[1]
    axis_hist = data["axis_only_similarity"]["histogram"]
    embed_hist = data["embedding_similarity"]["histogram"]
    edges = axis_hist["bin_edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    width = (edges[1] - edges[0]) * 0.4

    ax2.bar([c - width / 2 for c in centers], axis_hist["counts"], width=width, label="4축 단독", color="#2a78d6")
    ax2.bar([c + width / 2 for c in centers], embed_hist["counts"], width=width, label="임베딩", color="#eb6834")
    ax2.axvline(0.8, color="gray", linestyle="--", linewidth=1, label="유사도 0.8 기준선")
    ax2.set_title("영화쌍 코사인 유사도 분포 비교")
    ax2.set_xlabel("코사인 유사도")
    ax2.set_ylabel("쌍(pair) 개수")
    ax2.legend()

    stat_text = (
        f"4축 단독: 평균={data['axis_only_similarity']['mean']:.3f}, "
        f"0.8초과={data['axis_only_similarity']['pct_above_0.8']:.1f}%\n"
        f"임베딩:  평균={data['embedding_similarity']['mean']:.3f}, "
        f"0.8초과={data['embedding_similarity']['pct_above_0.8']:.1f}%"
    )
    fig.text(0.5, 0.01, stat_text, ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(BASE_DIR, "axis_distribution.png")
    plt.savefig(out_path, dpi=150)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
