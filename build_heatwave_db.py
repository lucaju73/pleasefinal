# -*- coding: utf-8 -*-
"""
서울시 독거노인 폭염 취약지역 분석용 SQLite DB 생성 스크립트
입력 파일:
- 독거노인+현황(연령별_동별)_20260527135709.csv
- 서울시 무더위쉼터.csv
- OBS_ASOS_TIM_20260527142057.csv  # 선택
출력 파일:
- heatwave_sql_final.db
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
ELDERLY_CSV = BASE / "독거노인+현황(연령별_동별)_20260527135709.csv"
SHELTER_CSV = BASE / "서울시 무더위쉼터.csv"
WEATHER_CSV = BASE / "OBS_ASOS_TIM_20260527142057.csv"
OUT_DB = BASE / "heatwave_sql_final.db"

DISTRICTS = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구"
]
TOP5_ORDER = ["중랑구", "강남구", "강북구", "노원구", "은평구"]


def read_csv_any(path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


def normalize_district(value):
    if pd.isna(value):
        return None
    s = str(value).strip().replace(" ", "")
    if s == "동대문":
        return "동대문구"
    if s in DISTRICTS:
        return s
    for d in DISTRICTS:
        if d.replace("구", "") == s:
            return d
    return s


def clean_elderly(path: Path):
    raw = read_csv_any(path)
    df = raw[pd.to_numeric(raw["2024"], errors="coerce").notna()].copy()
    rename = {
        "동별(1)": "sido", "동별(2)": "district", "동별(3)": "dong",
        "2024": "elderly_total", "2024.1": "elderly_65_79", "2024.2": "elderly_80_plus",
        "2024.3": "basic_livelihood_total", "2024.4": "basic_livelihood_65_79", "2024.5": "basic_livelihood_80_plus",
        "2024.6": "low_income_total", "2024.7": "low_income_65_79", "2024.8": "low_income_80_plus",
        "2024.9": "general_total", "2024.10": "general_65_79", "2024.11": "general_80_plus",
    }
    df = df.rename(columns=rename)[list(rename.values())]
    df["district"] = df["district"].apply(normalize_district)

    for col in list(rename.values())[3:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["elderly_80_plus_ratio"] = np.where(
        df["elderly_total"] > 0,
        (df["elderly_80_plus"] / df["elderly_total"] * 100).round(2),
        0,
    )

    elderly_district = df[(df["dong"] == "소계") & (df["district"].isin(DISTRICTS))].copy().reset_index(drop=True)
    elderly_dong = df[(df["dong"] != "소계") & (df["district"].isin(DISTRICTS))].copy().reset_index(drop=True)
    return elderly_district, elderly_dong


def extract_district(text):
    if pd.isna(text):
        return None
    s = str(text)
    match = re.search(r"서울특별시\s*([가-힣]+구)", s)
    if match:
        return normalize_district(match.group(1))
    for d in DISTRICTS:
        if d in s:
            return d
    return None


def extract_dong(text):
    if pd.isna(text):
        return None
    match = re.search(r"([가-힣0-9]+동)", str(text))
    return match.group(1) if match else None


def clean_shelters(path: Path):
    raw = read_csv_any(path)
    df = raw.rename(columns={
        "시설년도": "year", "위치코드": "location_code", "시설구분1": "facility_type1", "시설구분2": "facility_type2",
        "쉼터명칭": "shelter_name", "도로명주소": "road_address", "지번주소": "jibun_address",
        "시설면적": "area", "이용가능인원": "capacity", "비고": "memo", "경도": "longitude", "위도": "latitude",
        "X좌표(EPSG:5186)": "x_epsg5186", "Y좌표(EPSG:5186)": "y_epsg5186",
    }).copy()

    df = df[[
        "year", "location_code", "facility_type1", "facility_type2", "shelter_name", "road_address", "jibun_address",
        "area", "capacity", "memo", "longitude", "latitude", "x_epsg5186", "y_epsg5186"
    ]]
    df["district"] = df["road_address"].apply(extract_district)
    df.loc[df["district"].isna(), "district"] = df.loc[df["district"].isna(), "jibun_address"].apply(extract_district)
    df["dong_guess"] = df["jibun_address"].apply(extract_dong)

    for col in ["area", "capacity", "longitude", "latitude", "x_epsg5186", "y_epsg5186"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["capacity"] = df["capacity"].fillna(0).astype(int)
    df = df[df["district"].isin(DISTRICTS)].copy().reset_index(drop=True)

    return df[[
        "year", "location_code", "facility_type1", "facility_type2", "shelter_name", "district", "dong_guess",
        "road_address", "jibun_address", "area", "capacity", "longitude", "latitude", "x_epsg5186", "y_epsg5186"
    ]]


def clean_weather(path: Path):
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = read_csv_any(path)
    df = raw.rename(columns={
        "지점": "station_id", "지점명": "station_name", "일시": "datetime",
        "기온(°C)": "temperature_c", "지면온도(°C)": "ground_temperature_c",
    })[["station_id", "station_name", "datetime", "temperature_c", "ground_temperature_c"]]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").astype(str)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["ground_temperature_c"] = pd.to_numeric(df["ground_temperature_c"], errors="coerce")

    summary = pd.DataFrame([{
        "station_name": df["station_name"].dropna().iloc[0] if df["station_name"].notna().any() else "서울",
        "period_start": df["datetime"].min(),
        "period_end": df["datetime"].max(),
        "hourly_count": len(df),
        "avg_temperature_c": round(df["temperature_c"].mean(), 2),
        "max_temperature_c": round(df["temperature_c"].max(), 2),
        "min_temperature_c": round(df["temperature_c"].min(), 2),
        "avg_ground_temperature_c": round(df["ground_temperature_c"].mean(), 2),
        "max_ground_temperature_c": round(df["ground_temperature_c"].max(), 2),
    }])
    return df, summary


def build_db():
    elderly_district, elderly_dong = clean_elderly(ELDERLY_CSV)
    shelters = clean_shelters(SHELTER_CSV)
    weather_hourly, weather_summary = clean_weather(WEATHER_CSV)

    shelter_district = shelters.groupby("district", as_index=False).agg(
        shelter_count=("shelter_name", "count"),
        total_capacity=("capacity", "sum"),
        avg_capacity=("capacity", "mean"),
        total_area=("area", "sum"),
        avg_area=("area", "mean"),
        capacity_missing_count=("capacity", lambda x: int((x.fillna(0) == 0).sum())),
        area_missing_count=("area", lambda x: int(x.isna().sum())),
        public_facility_count=("facility_type1", lambda x: int((x == "공공시설").sum())),
        senior_facility_count=("facility_type2", lambda x: int(x.astype(str).str.contains("경로당|노인|복지", regex=True, na=False).sum())),
        avg_latitude=("latitude", "mean"),
        avg_longitude=("longitude", "mean"),
    )

    shelter_district = pd.DataFrame({"district": DISTRICTS}).merge(shelter_district, on="district", how="left")
    for col in ["shelter_count", "total_capacity", "avg_capacity", "total_area", "avg_area", "capacity_missing_count", "area_missing_count", "public_facility_count", "senior_facility_count", "avg_latitude", "avg_longitude"]:
        shelter_district[col] = shelter_district[col].fillna(0)
    for col in ["shelter_count", "total_capacity", "capacity_missing_count", "area_missing_count", "public_facility_count", "senior_facility_count"]:
        shelter_district[col] = shelter_district[col].astype(int)
    for col in ["avg_capacity", "total_area", "avg_area", "avg_latitude", "avg_longitude"]:
        shelter_district[col] = shelter_district[col].round(2)

    district_summary = elderly_district.merge(shelter_district, on="district", how="left")
    district_summary["elderly_80_plus_rate"] = district_summary["elderly_80_plus_ratio"]
    district_summary["vulnerable_elderly_rate"] = ((district_summary["basic_livelihood_total"] + district_summary["low_income_total"]) / district_summary["elderly_total"] * 100).round(2)
    district_summary["shelters_per_1000"] = (district_summary["shelter_count"] / district_summary["elderly_total"] * 1000).round(2)
    district_summary["elderly_per_shelter"] = np.where(district_summary["shelter_count"] > 0, district_summary["elderly_total"] / district_summary["shelter_count"], np.nan).round(2)
    district_summary["capacity_rate"] = (district_summary["total_capacity"] / district_summary["elderly_total"] * 100).round(2)

    district_summary = district_summary[[
        "district", "elderly_total", "elderly_65_79", "elderly_80_plus", "elderly_80_plus_rate",
        "basic_livelihood_total", "low_income_total", "general_total", "vulnerable_elderly_rate", "shelter_count",
        "total_capacity", "avg_capacity", "total_area", "avg_area", "capacity_missing_count", "area_missing_count",
        "public_facility_count", "senior_facility_count", "shelters_per_1000", "elderly_per_shelter", "capacity_rate"
    ]]

    thr_demand = district_summary["elderly_total"].quantile(.6)
    thr_access = district_summary["shelters_per_1000"].quantile(.4)
    thr_capacity = district_summary["capacity_rate"].quantile(.4)
    thr_senior = district_summary["elderly_80_plus_rate"].quantile(.6)

    priority = district_summary.copy()
    priority["demand_high"] = (priority["elderly_total"] >= thr_demand).astype(int)
    priority["access_low"] = (priority["shelters_per_1000"] <= thr_access).astype(int)
    priority["capacity_low"] = (priority["capacity_rate"] <= thr_capacity).astype(int)
    priority["senior_high"] = (priority["elderly_80_plus_rate"] >= thr_senior).astype(int)
    priority["priority_score"] = priority["demand_high"] * 3 + priority["access_low"] * 3 + priority["capacity_low"] * 3 + priority["senior_high"] * 1

    def priority_type(row):
        if row.demand_high and row.access_low and row.capacity_low:
            return "접근성 + 수용능력 동시 부족형"
        if row.demand_high and row.capacity_low and row.senior_high:
            return "수용능력 + 고령 취약성 부족형"
        if row.demand_high and row.capacity_low:
            return "수용능력 부족형"
        if row.demand_high and row.access_low:
            return "접근성 부족형"
        if row.senior_high:
            return "고령 취약성 주의지역"
        return "상대적 안정지역"

    priority["priority_type"] = priority.apply(priority_type, axis=1)
    district_priority = priority.drop(columns=["demand_high", "access_low", "capacity_low", "senior_high"])

    criteria_map = {
        "중랑구": "수요 높음 · 접근성 부족 · 수용능력 부족",
        "강남구": "수요 높음 · 접근성 부족 · 수용능력 부족",
        "강북구": "수요 높음 · 접근성 부족 · 수용능력 부족",
        "노원구": "수요 높음 · 수용능력 부족 · 고령 취약성 높음",
        "은평구": "수요 높음 · 수용능력 부족 · 고령 취약성 높음",
    }
    improve_map = {
        "중랑구": "신규 쉼터 지정 · 수용인원 확대",
        "강남구": "생활권 내 쉼터 확충 · 위치 안내 강화",
        "강북구": "공공시설 활용 · 수용공간 확대",
        "노원구": "기존 쉼터 수용인원 확대 · 고령층 안내",
        "은평구": "경로당·복지관 운영 강화 · 전화·방문 안내",
    }
    top5 = district_priority[district_priority["district"].isin(TOP5_ORDER)].copy()
    top5["rank"] = top5["district"].map({d: i + 1 for i, d in enumerate(TOP5_ORDER)})
    top5["criteria"] = top5["district"].map(criteria_map)
    top5["improvement_direction"] = top5["district"].map(improve_map)
    top5 = top5.sort_values("rank")

    factor_top5 = pd.DataFrame([{
        "district": d,
        "demand_high_score": 3,
        "access_low_score": 3 if d in ["중랑구", "강남구", "강북구"] else 0,
        "capacity_low_score": 3,
        "senior_high_score": 1 if d in ["노원구", "은평구"] else 0,
    } for d in TOP5_ORDER])
    factor_top5["total_score"] = factor_top5[["demand_high_score", "access_low_score", "capacity_low_score", "senior_high_score"]].sum(axis=1)

    if OUT_DB.exists():
        OUT_DB.unlink()
    conn = sqlite3.connect(OUT_DB)
    for name, frame in [
        ("elderly_district", elderly_district),
        ("elderly_dong", elderly_dong),
        ("shelters", shelters),
        ("weather_hourly", weather_hourly),
        ("weather_summary", weather_summary),
        ("shelter_district", shelter_district),
        ("district_summary", district_summary),
        ("district_priority", district_priority),
        ("top5_priority", top5),
        ("top5_factor_scores", factor_top5),
    ]:
        frame.to_sql(name, conn, index=False, if_exists="replace")

    cur = conn.cursor()
    for table in ["elderly_district", "elderly_dong", "shelters", "district_summary", "district_priority"]:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_district ON {table}(district)")
    conn.commit()
    conn.close()

    print(f"DB created: {OUT_DB}")
    print(f"district_summary: {len(district_summary)} rows")
    print(f"shelters: {len(shelters)} rows")
    print(f"elderly_dong: {len(elderly_dong)} rows")


if __name__ == "__main__":
    build_db()
