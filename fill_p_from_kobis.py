"""
A3-2: KOBIS 일별 박스오피스 API로 실제 흥행 데이터를 가져와 P축(대중성)을 정규화한다.

방식:
- 여러 날짜의 KOBIS 일별 박스오피스(top10)를 조회해서, 각 영화가 top10에 들었던
  날짜들 중 누적관객수(audiAcc)가 가장 큰(=가장 최근/완전한) 값을 채택
- 매칭된 영화의 box_office 필드를 갱신, status가 upcoming으로 잘못돼있던 영화는 now_showing으로 정정
- 누적관객수를 log10 스케일로 min-max 정규화 -> P(0~100)
  (원 관객수 그대로 min-max하면 대형 흥행작 하나가 전체를 다 찌그러뜨려서 log스케일 사용)
- 매칭 안 된 영화는 기존 LLM 추정 P 그대로 유지 (건드리지 않음)

KOBIS_API_KEY는 환경변수로만 읽는다 (코드에 절대 하드코딩하지 않음).
"""

import json
import math
import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 이 날짜들의 일별 박스오피스(top10)를 훑어서 우리 movie_db와 매칭한다.
# (각 영화의 개봉일 전후 + 최근일자를 섞어서, 이미 top10에서 내려간 소형 흥행작도 최대한 잡아냄)
SCAN_DATES = [
    "20260720", "20260713", "20260709", "20260708", "20260702", "20260527",
]


def fetch_daily_box_office(target_dt: str) -> list:
    key = os.environ["KOBIS_API_KEY"]
    url = (
        "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/"
        f"searchDailyBoxOfficeList.json?key={key}&targetDt={target_dt}"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["boxOfficeResult"]["dailyBoxOfficeList"]


def normalize_title(t: str) -> str:
    return t.replace(" ", "").replace(":", "")


def normalize_p_log_minmax(matched: dict) -> dict:
    """matched: {korean_title: audiAcc(int)} -> {korean_title: P(0~100)}"""
    log_values = {title: math.log10(max(audi, 1)) for title, audi in matched.items()}
    lo, hi = min(log_values.values()), max(log_values.values())
    span = hi - lo
    return {
        title: round((v - lo) / span * 100) if span > 0 else 50
        for title, v in log_values.items()
    }


def main():
    movies_path = os.path.join(BASE_DIR, "movie_db_scored.json")
    with open(movies_path, encoding="utf-8") as f:
        movies = json.load(f)

    norm_to_title = {normalize_title(m["korean_title"]): m["korean_title"] for m in movies}

    # 여러 날짜를 훑어서, 영화별로 누적관객수가 가장 큰(=가장 완전한) 항목을 채택
    best_by_title = {}
    for dt in SCAN_DATES:
        for item in fetch_daily_box_office(dt):
            key = normalize_title(item["movieNm"])
            if key not in norm_to_title:
                continue
            title = norm_to_title[key]
            audi = int(item["audiAcc"])
            if title not in best_by_title or audi > int(best_by_title[title]["audiAcc"]):
                best_by_title[title] = {**item, "targetDt": dt}

    print(f"KOBIS 매칭된 영화: {len(best_by_title)}편")

    audi_acc = {t: int(it["audiAcc"]) for t, it in best_by_title.items()}
    p_scores = normalize_p_log_minmax(audi_acc)

    title_to_movie = {m["korean_title"]: m for m in movies}
    for title, item in sorted(best_by_title.items(), key=lambda kv: -int(kv[1]["audiAcc"])):
        m = title_to_movie[title]

        if m["status"] == "upcoming":
            print(f"  status 정정: {title} upcoming -> now_showing")
            m["status"] = "now_showing"

        m["box_office"] = {
            "as_of": item["targetDt"],
            "rank": int(item["rank"]),
            "daily_audience": int(item["audiCnt"]),
            "cumulative_audience": int(item["audiAcc"]),
            "sales_share": float(item["salesShare"]),
        }

        old_p = m["mvti"]["axis_scores"]["P"]
        new_p = p_scores[title]
        m["mvti"]["axis_scores"]["P"] = new_p
        m["mvti"]["reasons"]["P"] = (
            f"KOBIS 박스오피스 {item['rank']}위, "
            f"누적 관객 {int(item['audiAcc']):,}명 (log 정규화, {item['targetDt']} 기준) "
            f"[기존 추정치 {old_p} -> {new_p}]"
        )
        print(f"  {title}: P {old_p} -> {new_p} (누적 {int(item['audiAcc']):,}명, {item['rank']}위, {item['targetDt']})")

    with open(movies_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\nmovie_db_scored.json 갱신 완료")


if __name__ == "__main__":
    main()
