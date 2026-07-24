"""
TMDB 영화 정보 수집 스크립트
CGV Campus Crew · Chat GVt

사용법:
    pip install requests
    python fetch_tmdb.py

결과물:
    movie_db.json  - 다음 단계(LLM 벡터 생성)에서 그대로 읽어 쓰기 좋은 형태
    movie_db.csv   - 엑셀에서 바로 열어보기용
"""

import csv
import json
import os
import time
from typing import Optional

import requests

# 실제 배포/공유 시에는 하드코딩 대신 환경변수(TMDB_API_KEY)로 옮기는 걸 권장
API_KEY = os.environ.get("TMDB_API_KEY", "")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w342"
REQUEST_DELAY = 0.15  # 초 단위. TMDB rate limit(초당 다건 요청 제한) 대응용

MOVIES: list[tuple[str, str]] = [
    ("모아나", "상영중"), ("토이 스토리 5", "상영중"), ("눈동자", "상영중"),
    ("블리치 천년혈전 편- 화진담", "상영중"), ("다윗", "상영중"), ("마티 슈프림", "상영중"),
    ("군체", "상영중"), ("시크릿 에이전트", "상영중"), ("극장판 암살교실 모두의 시간", "상영중"),
    ("하나 코리아", "상영중"), ("키퍼", "상영중"), ("회로", "상영중"), ("와일드 씽", "상영중"),
    ("해피엔드", "상영중"), ("백룸", "상영중"), ("너바나 더 밴드", "상영중"),
    ("와인드업- 더 무비", "상영중"), ("레디 오어 낫- 죽음의 숨바꼭질", "상영중"),
    ("싱 스트리트", "상영중"), ("훌라 걸즈", "상영중"),

    ("경멸", "상영예정"), ("가능주의자", "상영예정"), ("지구 최후의 여자", "상영예정"),
    ("큐어", "상영예정"), ("호프", "상영예정"), ("미니언즈 & 몬스터즈", "상영예정"),
    ("파리의 사생활", "상영예정"), ("지느러미", "상영예정"), ("뱀의 길", "상영예정"),
    ("소영의 노력", "상영예정"), ("예스! 유 캔", "상영예정"), ("드림 애니멀즈-더무비", "상영예정"),
    ("스파이더맨 - 브랜드 뉴 데이", "상영예정"), ("불멸의 존재들- 이집트 박물관의 경이들", "상영예정"),
    ("어떻게 해야 했을까", "상영예정"), ("산양들", "상영예정"), ("삼국지 - 관도대전", "상영예정"),
    ("창극 패왕별희", "상영예정"), ("오디세이", "상영예정"), ("사랑의 하츄핑- 고래보석의 전설", "상영예정"),
    ("사자의 서", "상영예정"),

    ("비포 선라이즈", "기타"), ("퍼시픽션", "기타"), ("피아니스트", "기타"), ("미명", "기타"),
    ("가족여행", "기타"), ("고독의 오후", "기타"), ("비발디와 나", "기타"),
    ("소리없이 나빌레라", "기타"), ("쇼타씨의 마지막 출장", "기타"), ("여름의 카메라", "기타"),
    ("충충충", "기타"), ("침묵의 친구", "기타"), ("순례자들은 왜 돌아오지 않는가", "기타"),
    ("신사-악귀의 속삭임", "기타"),
]

def tmdb_get(path: str, **params) -> dict:
    """TMDB API GET 요청. 429(rate limit)면 한 번 대기 후 재시도."""
    params["api_key"] = API_KEY
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=10)
    if resp.status_code == 429:
        time.sleep(1.5)
        resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def search_movie(title: str) -> Optional[dict]:
    """한국어 검색 우선, 결과 없으면 원어 검색으로 재시도해서 최상위(인기도순) 결과 반환."""
    data = tmdb_get("/search/movie", query=title, language="ko-KR", include_adult=False)
    results = data.get("results") or []
    if not results:
        data = tmdb_get("/search/movie", query=title, include_adult=False)
        results = data.get("results") or []
    return results[0] if results else None


def fetch_movie_detail(movie_id: int) -> dict:
    """장르·러닝타임·줄거리·감독·출연진·키워드를 한 번에 조회 (append_to_response 활용)."""
    detail = tmdb_get(f"/movie/{movie_id}", language="ko-KR", append_to_response="credits,keywords")
    if not detail.get("overview"):
        # 한국어 줄거리가 비어있으면 영어 줄거리로 대체
        detail_en = tmdb_get(f"/movie/{movie_id}", append_to_response="credits")
        detail["overview"] = detail_en.get("overview")
    return detail


def build_record(query_title: str, status: str) -> dict:
    record = {
        "구분": status,
        "검색_제목": query_title,
        "tmdb_id": None,
        "tmdb_제목": None,
        "원제": None,
        "개봉일": None,
        "장르": None,
        "러닝타임": None,
        "줄거리": None,
        "감독": None,
        "출연진": None,
        "키워드": None,
        "포스터_url": None,
        "tmdb_url": None,
        "매칭_상태": "검색결과 없음",
    }
    try:
        candidate = search_movie(query_title)
        time.sleep(REQUEST_DELAY)
        if not candidate:
            return record

        detail = fetch_movie_detail(candidate["id"])
        time.sleep(REQUEST_DELAY)

        crew = (detail.get("credits") or {}).get("crew", [])
        cast = (detail.get("credits") or {}).get("cast", [])
        keywords = (detail.get("keywords") or {}).get("keywords", [])

        record.update({
            "tmdb_id": detail.get("id"),
            "tmdb_제목": detail.get("title"),
            "원제": detail.get("original_title"),
            "개봉일": detail.get("release_date") or None,
            "장르": ", ".join(g["name"] for g in detail.get("genres", [])) or None,
            "러닝타임": f"{detail['runtime']}분" if detail.get("runtime") else None,
            "줄거리": detail.get("overview") or None,
            "감독": ", ".join(c["name"] for c in crew if c.get("job") == "Director") or None,
            "출연진": ", ".join(c["name"] for c in cast[:6]) or None,
            "키워드": ", ".join(k["name"] for k in keywords) or None,
            "포스터_url": f"{IMG_BASE}{detail['poster_path']}" if detail.get("poster_path") else None,
            "tmdb_url": f"https://www.themoviedb.org/movie/{detail.get('id')}",
            "매칭_상태": "매칭됨",
        })
    except requests.RequestException as e:
        record["매칭_상태"] = f"오류: {e}"
    return record


def main():
    records = []
    total = len(MOVIES)
    for i, (title, status) in enumerate(MOVIES, 1):
        print(f"[{i}/{total}] {title} 조회 중...")
        record = build_record(title, status)
        records.append(record)
        mark = "✓" if record["매칭_상태"] == "매칭됨" else "✗"
        print(f"    {mark} {record['매칭_상태']}")

    with open("movie_db.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open("movie_db.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    found = sum(1 for r in records if r["매칭_상태"] == "매칭됨")
    print(f"\n완료: {found}/{total}건 매칭 → movie_db.json, movie_db.csv 저장됨")


if __name__ == "__main__":
    main()