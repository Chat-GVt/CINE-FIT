"""
B2: movie_db_scored.json의 overview(줄거리) + keywords(키워드)를 BGE-M3로 임베딩.

이 환경(Microsoft Store Python)은 torch(PyTorch) 네이티브 DLL 로딩이 샌드박스에 막혀서
sentence-transformers/torch 경로가 아니라 onnxruntime + BAAI/bge-m3 공식 ONNX 파일을
직접 사용한다 (BAAI/bge-m3 저장소에 onnx/model.onnx가 이미 공식 제공됨).

출력: movie_embeddings.json  {movie_id: [1024차원 float 벡터]}
"""

import json
import os

from bge_embedder import load_model, embed_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    movies_path = os.path.join(BASE_DIR, "movie_db_scored.json")
    out_path = os.path.join(BASE_DIR, "movie_embeddings.json")

    with open(movies_path, encoding="utf-8") as f:
        movies = json.load(f)

    print("BGE-M3 ONNX 모델 로딩 중...")
    load_model()

    embeddings = {}
    for i, m in enumerate(movies, 1):
        text = m["overview"] + " " + " ".join(m.get("keywords", []))
        vec = embed_text(text)
        embeddings[m["movie_id"]] = vec
        print(f"  [{i}/{len(movies)}] {m['korean_title']} -> {len(vec)}차원")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"\n{out_path} 저장 완료 ({len(embeddings)}편)")


if __name__ == "__main__":
    main()
