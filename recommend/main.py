"""
CINE:FIT 추천 파이프라인 실행 엔트리포인트.

    python main.py

흐름:
  1. 데이터 로드 (영화 DB / 설문 문항 / 16유형 / 사용자 입력)
     + 영화 임베딩은 캐시 확인 -> 없거나 갱신됐으면 자동 생성
  2. 사용자 MVTI 계산      -> engine.build_user_profile
  3. 자연어 취향 임베딩     -> dataset.embed_text
  4. 유사도 매칭 + 점수 계산 -> engine.recommend  (축 0.7 : 임베딩 0.3)
  5. 유형 설명 + 추천 근거  -> engine.generate_result_copy (OpenAI 1회 호출)
"""

from dotenv import load_dotenv
load_dotenv()

from config import TOP_N, W_AXIS, W_EMBED
from dataset import (
    embed_text,
    get_movie_embeddings,
    load_movies,
    load_questions,
    load_types,
    load_user_input,
)
from engine import build_user_profile, generate_result_copy, recommend


def print_profile(profile, type_description: str) -> None:
    print("\n" + "=" * 60)
    print(f"STEP 1. {profile.type_code} {profile.type_name}")
    print("=" * 60)
    print(f"  키워드: {' · '.join(profile.keywords)}")
    for axis, r in profile.ratios.items():
        print(f"  {axis}축: {r['label']}({r['pole']}) {r['ratio']}%  (원점수 {profile.axis_scores[axis]})")
    print("  대표 캐릭터: " + ", ".join(f"{c['name']}({c['work']})" for c in profile.characters))
    print(f"\n  {type_description}")


def print_recommendations(recommendations, reasons) -> None:
    print("\n" + "=" * 60)
    print(f"STEP 2. 추천 결과 TOP {len(recommendations)}")
    print("=" * 60)
    for rank, rec in enumerate(recommendations, 1):
        w_axis, w_embed = rec.weights
        print(f"\n[{rank}위] {rec.title}  —  {rec.total_score}점")
        print(f"  장르: {', '.join(rec.movie.get('genre', []))}")
        print(
            f"  세부: MVTI 유사도 {rec.axis_score} (x{w_axis:.1f}) | "
            f"임베딩 유사도 {rec.embed_score} (x{w_embed:.1f})"
        )
        print(f"  추천 이유: {reasons[rec.movie['movie_id']]}")


def main():
    # 1. 데이터 로드
    movies = load_movies()
    questions = load_questions()
    types = load_types()
    user_input = load_user_input()
    print(f"영화 {len(movies)}편 / 문항 {len(questions)}개 / 유형 {len(types)}개 로드")

    movie_embeddings = get_movie_embeddings(movies)

    # 2. 사용자 MVTI 계산
    profile = build_user_profile(
        questions=questions,
        answers=user_input["answers"],
        types=types,
        natural_language=user_input.get("natural_language"),
    )

    # 3. 자연어 취향 임베딩
    user_embedding = None
    if profile.natural_language:
        print("\nKURE로 자연어 취향 임베딩 중...")
        user_embedding = embed_text(profile.natural_language)

    # 4. 유사도 매칭 + 최종 점수
    recommendations = recommend(
        profile=profile,
        movies=movies,
        movie_embeddings=movie_embeddings,
        user_embedding=user_embedding,
        top_n=TOP_N,
    )
    w_axis, w_embed = recommendations[0].weights
    print(f"적용 가중치: 축 {w_axis:.1f} : 임베딩 {w_embed:.1f} (설정값 {W_AXIS}:{W_EMBED})")

    # 5. 유형 설명 + 추천 근거 생성 (OpenAI 1회 호출)
    copy = generate_result_copy(profile, recommendations)
    reasons = copy["reasons"]

    print_profile(profile, copy["type_description"])
    print_recommendations(recommendations, reasons)

    # 프론트로 넘길 결과 payload
    return {
        **profile.as_dict(),
        "type_description": copy["type_description"],
        "movies": [
            {**rec.as_dict(), "reason": reasons[rec.movie["movie_id"]]}
            for rec in recommendations
        ],
    }


if __name__ == "__main__":
    main()
