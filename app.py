import os
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# 기본 설정
# ==========================================================
st.set_page_config(
    page_title="서울시 독거노인 폭염 취약지역 분석",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path("heatwave_sql_final.db")
GEOJSON_PATH = Path("seoul_district_boundary_simplified.geojson")
HEAT_ILLNESS_PATH = Path("heat_illness.csv")

TOP5_ORDER = ["중랑구", "강남구", "강북구", "노원구", "은평구"]

# 기준 점수 설명
SCORE_RULES = {
    "수요 높음": "독거노인 수 상위 40%",
    "접근성 부족": "독거노인 1,000명당 쉼터 수 하위 40%",
    "수용능력 부족": "쉼터 수용률 하위 40%",
    "고령 취약성 높음": "80세 이상 독거노인 비율 상위 40%",
}

# ==========================================================
# 스타일
# ==========================================================
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background: #f8fafc;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        height: 100%;
    }
    .big-message {
        background: linear-gradient(90deg, #fff7ed, #fef3c7);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid #fed7aa;
        font-weight: 600;
    }
    .small-note {font-size: 0.85rem; color: #64748b;}
    .section-box {
        background: #ffffff;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# 데이터 로드 함수
# ==========================================================
@st.cache_data(show_spinner=False)
def read_table(table_name: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(f"SELECT * FROM {table_name}", con)

@st.cache_data(show_spinner=False)
def load_geojson():
    if not GEOJSON_PATH.exists():
        return None
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_heat_illness() -> pd.DataFrame:
    """선택 파일 heat_illness.csv가 있을 때만 읽는다."""
    if not HEAT_ILLNESS_PATH.exists():
        return pd.DataFrame()
    encodings = ["utf-8-sig", "utf-8", "cp949"]
    for enc in encodings:
        try:
            return pd.read_csv(HEAT_ILLNESS_PATH, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_thresholds(df: pd.DataFrame):
    """상위·하위 40% 분위 기준 계산"""
    return {
        "elderly_high": df["elderly_total"].quantile(0.60),
        "shelters_low": df["shelters_per_1000"].quantile(0.40),
        "capacity_low": df["capacity_rate"].quantile(0.40),
        "senior_high": df["elderly_80_plus_rate"].quantile(0.60),
    }

# ==========================================================
# 기본 데이터
# ==========================================================
district_summary = read_table("district_summary")
district_priority = read_table("district_priority")
top5_priority = read_table("top5_priority")
top5_factor_scores = read_table("top5_factor_scores")
elderly_dong = read_table("elderly_dong")
shelters = read_table("shelters")
weather_hourly = read_table("weather_hourly")
weather_summary = read_table("weather_summary")
geojson = load_geojson()
heat_illness = load_heat_illness()

# 앱이 실행은 되되, 핵심 파일이 없으면 안내
st.sidebar.title("☀️ 폭염 취약지역 분석")
st.sidebar.caption("서울시 독거노인 수요와 무더위쉼터 공급 비교")

if not DB_PATH.exists():
    st.error("`heatwave_sql_final.db` 파일이 없습니다. GitHub 최상위 폴더에 DB 파일을 올려주세요.")
    st.stop()

# ==========================================================
# 공통 차트 함수
# ==========================================================
def apply_plot_layout(fig, height=520):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=30),
        font=dict(family="Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def format_top5_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank", "district", "criteria", "improvement_direction"]
    table = df[cols].copy()
    table.columns = ["순위", "자치구", "해당 취약 기준", "개선 방향"]
    return table

# ==========================================================
# 타이틀
# ==========================================================
st.title("서울시 독거노인 폭염 취약지역 분석")
st.caption("무더위쉼터 개선 우선지역 도출 대시보드")

if not district_summary.empty:
    total_elderly = int(district_summary["elderly_total"].sum())
    total_shelters = int(district_summary["shelter_count"].sum())
    total_capacity = int(district_summary["total_capacity"].sum())
    avg_capacity_rate = district_summary["capacity_rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 자치구", f"{district_summary['district'].nunique()}개")
    c2.metric("독거노인 수", f"{total_elderly:,}명")
    c3.metric("무더위쉼터 수", f"{total_shelters:,}개")
    c4.metric("평균 쉼터 수용률", f"{avg_capacity_rate:.1f}%")

# ==========================================================
# 탭 구성
# ==========================================================
tab_overview, tab_heat, tab_demand, tab_supply, tab_mismatch, tab_top5, tab_policy = st.tabs([
    "1. 개요",
    "2. 온열질환 데이터",
    "3. 독거노인 수요 분석",
    "4. 무더위쉼터 공급 분석",
    "5. 수요-공급 불일치 분석",
    "6. 개선 우선지역 TOP 5",
    "7. 개선 방안",
])

# ==========================================================
# 1. 개요
# ==========================================================
with tab_overview:
    st.header("1. 분석 개요")
    st.markdown(
        """
        <div class="big-message">
        핵심 질문: 서울시 독거노인이 많이 거주하는 지역에 무더위쉼터가 충분히 배치되어 있는가?
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("문제의식")
        st.markdown(
            """
            폭염은 단순히 더운 날씨가 아니라 폭염 취약계층에게 큰 부담이 될 수 있습니다.  
            특히 노인은 체온 조절 능력 저하, 만성질환, 이동 제약 등으로 폭염에 취약하며,  
            독거노인은 위기 상황에서 주변 도움을 받기 어렵고 발견·대응이 늦어질 위험이 있습니다.

            따라서 본 분석은 **독거노인을 폭염 대응 인프라 분석의 핵심 취약계층**으로 설정했습니다.
            """
        )
        st.info("문제는 쉼터가 있느냐가 아니라, 필요한 곳에 충분히 있느냐이다.")

    with right:
        st.subheader("분석 질문")
        st.markdown(
            """
            1. 독거노인 수요가 높은 자치구는 어디인가?  
            2. 그 지역의 쉼터 수와 수용가능인원은 충분한가?  
            3. 개선이 우선적으로 필요한 자치구는 어디인가?  
            4. 개선 우선지역은 어떤 유형으로 나뉘는가?
            """
        )

    st.divider()
    st.subheader("사용 데이터와 전처리 흐름")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="section-box">
        <b>원자료</b><br><br>
        · 서울시 독거노인 현황<br>
        · 서울시 무더위쉼터 현황<br>
        · SGIS 센서스 공간정보<br>
        · 기상 관측 데이터
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="section-box">
        <b>전처리</b><br><br>
        · 자치구 기준 통일<br>
        · 주소에서 자치구명 정리<br>
        · 수치형 컬럼 변환<br>
        · SHP → GeoJSON 변환
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="section-box">
        <b>SQL 분석</b><br><br>
        · GROUP BY: 자치구별 집계<br>
        · JOIN: 수요·공급 결합<br>
        · CASE WHEN: 유형 분류<br>
        · ORDER BY: TOP 5 도출
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="section-box">
        <b>최종 지표</b><br><br>
        · 독거노인 수<br>
        · 1,000명당 쉼터 수<br>
        · 쉼터 수용률<br>
        · 80세 이상 비율<br>
        · 개선 우선순위 점수
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("개선 우선지역 판단 기준")
    rule_cols = st.columns(4)
    for col, (name, desc) in zip(rule_cols, SCORE_RULES.items()):
        with col:
            st.metric(name, desc)
    st.caption("본 분석의 분위 기준: 상위·하위 40%")

# ==========================================================
# 2. 온열질환 데이터
# ==========================================================
with tab_heat:
    st.header("2. 온열질환 데이터")
    st.markdown("폭염이 단순한 더위가 아니라 건강 피해로 이어질 수 있다는 문제의식을 보여주는 보조 탭입니다.")

    if not heat_illness.empty:
        st.success("heat_illness.csv 파일을 불러왔습니다.")
        st.dataframe(heat_illness.head(20), use_container_width=True)

        cols = heat_illness.columns.tolist()
        numeric_cols = heat_illness.select_dtypes(include=np.number).columns.tolist()
        if len(cols) >= 2 and numeric_cols:
            x_col = st.selectbox("X축 컬럼", cols, index=0, key="heat_x")
            y_col = st.selectbox("Y축 컬럼", numeric_cols, index=0, key="heat_y")
            fig = px.bar(heat_illness, x=x_col, y=y_col, title="온열질환 데이터 시각화")
            st.plotly_chart(apply_plot_layout(fig, 460), use_container_width=True)
        else:
            st.warning("heat_illness.csv의 컬럼 구조를 자동으로 인식하기 어렵습니다. 표 형태로 확인해주세요.")
    else:
        st.warning("`heat_illness.csv` 파일이 없어서 온열질환 CSV 그래프는 표시되지 않습니다. 파일을 추가하면 이 탭에서 자동으로 시각화됩니다.")

    if not weather_hourly.empty:
        st.subheader("서울 관측 기온 변화")
        weather_hourly["datetime"] = pd.to_datetime(weather_hourly["datetime"], errors="coerce")
        fig = px.line(
            weather_hourly,
            x="datetime",
            y=["temperature_c", "ground_temperature_c"],
            labels={"value": "기온(℃)", "datetime": "시간", "variable": "구분"},
            title="서울 시간대별 기온 및 지면온도",
        )
        fig.for_each_trace(lambda t: t.update(name={"temperature_c": "기온", "ground_temperature_c": "지면온도"}.get(t.name, t.name)))
        st.plotly_chart(apply_plot_layout(fig, 480), use_container_width=True)

    if not weather_summary.empty:
        st.subheader("기상 요약")
        summary = weather_summary.copy()
        st.dataframe(summary, use_container_width=True)

# ==========================================================
# 3. 독거노인 수요 분석
# ==========================================================
with tab_demand:
    st.header("3. 독거노인 수요 분석")
    st.markdown("독거노인 수는 폭염 대응 인프라의 수요를 보여주는 핵심 지표입니다.")

    col1, col2 = st.columns(2)
    with col1:
        top_elderly = district_summary.sort_values("elderly_total", ascending=False).head(10)
        fig = px.bar(
            top_elderly.sort_values("elderly_total"),
            x="elderly_total",
            y="district",
            orientation="h",
            text="elderly_total",
            title="독거노인 수 상위 10개 자치구",
            labels={"elderly_total": "독거노인 수(명)", "district": "자치구"},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(apply_plot_layout(fig, 520), use_container_width=True)

    with col2:
        top_senior = district_summary.sort_values("elderly_80_plus_rate", ascending=False).head(10)
        fig = px.bar(
            top_senior.sort_values("elderly_80_plus_rate"),
            x="elderly_80_plus_rate",
            y="district",
            orientation="h",
            text="elderly_80_plus_rate",
            title="80세 이상 독거노인 비율 상위 10개 자치구",
            labels={"elderly_80_plus_rate": "80세 이상 비율(%)", "district": "자치구"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(apply_plot_layout(fig, 520), use_container_width=True)

    if not elderly_dong.empty:
        st.subheader("행정동별 독거노인 수 TOP 10")
        dong_col = "dong" if "dong" in elderly_dong.columns else elderly_dong.columns[0]
        district_col = "district" if "district" in elderly_dong.columns else None
        dong_df = elderly_dong[elderly_dong[dong_col] != "소계"].copy()
        dong_top = dong_df.sort_values("elderly_total", ascending=False).head(10)
        if district_col:
            dong_top["label"] = dong_top["district"] + " " + dong_top[dong_col]
        else:
            dong_top["label"] = dong_top[dong_col]
        fig = px.bar(
            dong_top.sort_values("elderly_total"),
            x="elderly_total",
            y="label",
            orientation="h",
            text="elderly_total",
            title="행정동별 독거노인 수 TOP 10",
            labels={"elderly_total": "독거노인 수(명)", "label": "행정동"},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(apply_plot_layout(fig, 500), use_container_width=True)

# ==========================================================
# 4. 무더위쉼터 공급 분석
# ==========================================================
with tab_supply:
    st.header("4. 무더위쉼터 공급 분석")
    st.markdown("쉼터의 단순 개수뿐 아니라 독거노인 규모 대비 접근성, 수용능력, 쉼터 부담을 함께 확인합니다.")

    col1, col2 = st.columns(2)
    with col1:
        low_access = district_summary.sort_values("shelters_per_1000", ascending=True).head(10)
        fig = px.bar(
            low_access.sort_values("shelters_per_1000", ascending=False),
            x="shelters_per_1000",
            y="district",
            orientation="h",
            text="shelters_per_1000",
            title="독거노인 1,000명당 쉼터 수 하위 10개 자치구",
            labels={"shelters_per_1000": "1,000명당 쉼터 수(개)", "district": "자치구"},
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(apply_plot_layout(fig, 520), use_container_width=True)

    with col2:
        low_capacity = district_summary.sort_values("capacity_rate", ascending=True).head(10)
        fig = px.bar(
            low_capacity.sort_values("capacity_rate", ascending=False),
            x="capacity_rate",
            y="district",
            orientation="h",
            text="capacity_rate",
            title="쉼터 수용률 하위 10개 자치구",
            labels={"capacity_rate": "쉼터 수용률(%)", "district": "자치구"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(apply_plot_layout(fig, 520), use_container_width=True)

    st.subheader("쉼터 1개당 담당 독거노인 수")
    burden = district_summary.sort_values("elderly_per_shelter", ascending=False).head(10)
    fig = px.bar(
        burden.sort_values("elderly_per_shelter"),
        x="elderly_per_shelter",
        y="district",
        orientation="h",
        text="elderly_per_shelter",
        title="쉼터 1개당 담당 독거노인 수 상위 10개 자치구",
        labels={"elderly_per_shelter": "쉼터 1개당 독거노인 수(명)", "district": "자치구"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(apply_plot_layout(fig, 520), use_container_width=True)

    if not shelters.empty and {"latitude", "longitude"}.issubset(shelters.columns):
        st.subheader("무더위쉼터 위치 분포")
        map_df = shelters.dropna(subset=["latitude", "longitude"]).copy()
        if len(map_df) > 0:
            sample_size = min(2500, len(map_df))
            map_df = map_df.sample(sample_size, random_state=42) if len(map_df) > sample_size else map_df
            fig = px.scatter_mapbox(
                map_df,
                lat="latitude",
                lon="longitude",
                hover_name="shelter_name" if "shelter_name" in map_df.columns else None,
                hover_data=[c for c in ["district", "capacity", "facility_type1"] if c in map_df.columns],
                zoom=10,
                height=620,
                title="서울시 무더위쉼터 위치 분포",
            )
            fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("좌표가 있는 쉼터를 지도에 표시했습니다. 표시 속도를 위해 필요 시 일부 표본만 표시됩니다.")

# ==========================================================
# 5. 수요-공급 불일치 분석
# ==========================================================
with tab_mismatch:
    st.header("5. 수요-공급 불일치 분석")
    st.markdown("발표문의 핵심 시각화인 기준선 산점도와 버블 산점도를 통해 수요와 공급의 불일치를 확인합니다.")

    thresholds = get_thresholds(district_summary)

    st.subheader("개선 우선지역 판단 기준 산점도")
    fig = px.scatter(
        district_summary,
        x="elderly_total",
        y="capacity_rate",
        text="district",
        size="shelter_count",
        hover_name="district",
        hover_data={
            "elderly_total": ":,",
            "capacity_rate": ":.2f",
            "shelter_count": ":,",
            "shelters_per_1000": ":.2f",
            "elderly_80_plus_rate": ":.2f",
        },
        title="독거노인 수 × 쉼터 수용률: 상위·하위 40% 기준선",
        labels={"elderly_total": "독거노인 수(명)", "capacity_rate": "쉼터 수용률(%)", "shelter_count": "쉼터 수"},
    )
    fig.add_vline(x=thresholds["elderly_high"], line_width=2, line_dash="dash", line_color="gray", annotation_text="독거노인 수 상위 40% 기준")
    fig.add_hline(y=thresholds["capacity_low"], line_width=2, line_dash="dash", line_color="gray", annotation_text="수용률 하위 40% 기준")
    fig.add_annotation(
        x=thresholds["elderly_high"] * 1.14,
        y=max(district_summary["capacity_rate"].min() + 1, thresholds["capacity_low"] * 0.55),
        text="수요 높음 + 수용능력 부족<br>개선 우선 검토 영역",
        showarrow=False,
        bgcolor="rgba(255, 237, 213, 0.85)",
        bordercolor="#fb923c",
        borderwidth=1,
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(apply_plot_layout(fig, 650), use_container_width=True)

    st.subheader("독거노인 수요와 쉼터 수용능력 버블 산점도")
    fig = px.scatter(
        district_summary,
        x="elderly_total",
        y="total_capacity",
        size="shelter_count",
        text="district",
        hover_name="district",
        hover_data={
            "elderly_total": ":,",
            "total_capacity": ":,",
            "shelter_count": ":,",
            "capacity_rate": ":.2f",
            "shelters_per_1000": ":.2f",
        },
        title="독거노인 수요와 무더위쉼터 총 수용가능인원 비교",
        labels={
            "elderly_total": "독거노인 수(명)",
            "total_capacity": "총 수용가능인원(명)",
            "shelter_count": "쉼터 수",
        },
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(apply_plot_layout(fig, 650), use_container_width=True)
    st.info("핵심 해석: 독거노인 수가 많다고 해서 반드시 수용가능인원도 높은 것은 아닙니다. 따라서 쉼터 개수뿐 아니라 실제 수용능력을 함께 봐야 합니다.")

# ==========================================================
# 6. 개선 우선지역 TOP 5
# ==========================================================
with tab_top5:
    st.header("6. 개선 우선지역 TOP 5")
    st.markdown("앞서 설정한 4개 기준을 종합해 최종 개선 우선지역을 도출했습니다.")

    if geojson is not None:
        fig = px.choropleth_mapbox(
            district_priority,
            geojson=geojson,
            locations="district",
            featureidkey="properties.district",
            color="priority_score",
            hover_name="district",
            hover_data={
                "priority_score": True,
                "elderly_total": ":,",
                "shelters_per_1000": ":.2f",
                "capacity_rate": ":.2f",
                "elderly_80_plus_rate": ":.2f",
            },
            mapbox_style="carto-positron",
            center={"lat": 37.5665, "lon": 126.9780},
            zoom=9.4,
            opacity=0.72,
            color_continuous_scale="YlOrRd",
            title="서울시 자치구별 최종 개선 우선순위 지도",
            labels={"priority_score": "개선 우선순위 점수"},
        )
        fig.update_layout(height=680, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("노란색·붉은색에 가까울수록 종합 점수가 높고, 개선 우선순위가 높은 자치구입니다.")
    else:
        st.warning("`seoul_district_boundary_simplified.geojson` 파일이 없어 지도는 표시되지 않습니다.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("자치구별 개선 우선순위 점수")
        score_rank = district_priority.sort_values("priority_score", ascending=False)
        fig = px.bar(
            score_rank.sort_values("priority_score"),
            x="priority_score",
            y="district",
            orientation="h",
            text="priority_score",
            title="개선 우선순위 점수 순위",
            labels={"priority_score": "우선순위 점수", "district": "자치구"},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(apply_plot_layout(fig, 620), use_container_width=True)

    with col2:
        st.subheader("최종 개선 우선지역 TOP 5")
        if not top5_priority.empty:
            st.dataframe(format_top5_table(top5_priority), use_container_width=True, hide_index=True)
        else:
            manual_top5 = district_priority[district_priority["district"].isin(TOP5_ORDER)].copy()
            manual_top5["rank"] = manual_top5["district"].apply(lambda x: TOP5_ORDER.index(x) + 1)
            st.dataframe(manual_top5.sort_values("rank"), use_container_width=True, hide_index=True)

    st.subheader("TOP 5 취약 원인 구성 그래프")
    factor_label_map = {
        "demand_high_score": "수요 높음",
        "access_low_score": "접근성 부족",
        "capacity_low_score": "수용능력 부족",
        "senior_high_score": "고령 취약성 높음",
    }
    if not top5_factor_scores.empty:
        factor_df = top5_factor_scores.copy()
        factor_df["district"] = pd.Categorical(factor_df["district"], categories=TOP5_ORDER, ordered=True)
        factor_df = factor_df.sort_values("district")
        long_df = factor_df.melt(
            id_vars=["district", "total_score"],
            value_vars=list(factor_label_map.keys()),
            var_name="factor",
            value_name="score",
        )
        long_df["factor"] = long_df["factor"].map(factor_label_map)
        long_df = long_df[long_df["score"] > 0]
        fig = px.bar(
            long_df,
            x="score",
            y="district",
            color="factor",
            orientation="h",
            text="score",
            title="개선 우선지역 TOP 5 취약 요인 구성",
            labels={"score": "취약 요인 점수", "district": "자치구", "factor": "취약 요인"},
        )
        fig.update_layout(barmode="stack")
        st.plotly_chart(apply_plot_layout(fig, 600), use_container_width=True)

    st.info("TOP 5는 모두 같은 이유로 취약한 것이 아니라, 4개 기준의 조합에 따라 ‘접근성+수용능력 동시 부족형’과 ‘수용능력+고령 취약성 부족형’으로 나뉩니다.")

# ==========================================================
# 7. 개선 방안
# ==========================================================
with tab_policy:
    st.header("7. 개선 방안")
    st.markdown("개선 우선지역은 부족한 기준이 다르므로, 같은 방식이 아니라 취약 기준별 맞춤 대응이 필요합니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="section-box">
        <h4>접근성 부족 포함 지역</h4>
        <b>대상</b><br>
        중랑구 · 강남구 · 강북구<br><br>
        <b>문제</b><br>
        독거노인 수 대비 쉼터 수 부족<br><br>
        <b>제안</b><br>
        · 신규 무더위쉼터 지정<br>
        · 공공시설·주민센터·복지관 활용<br>
        · 생활권 내 분산 배치<br>
        · 쉼터 위치 안내 강화
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="section-box">
        <h4>수용능력 부족 포함 지역</h4>
        <b>대상</b><br>
        중랑구 · 강남구 · 강북구 · 노원구 · 은평구<br><br>
        <b>문제</b><br>
        독거노인 수 대비 총 수용가능인원 부족<br><br>
        <b>제안</b><br>
        · 기존 쉼터 수용인원 확대<br>
        · 경로당·복지관 운영 강화<br>
        · 폭염특보 시 운영시간 보완<br>
        · 수용공간 보완
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="section-box">
        <h4>고령 취약성 높은 지역</h4>
        <b>대상</b><br>
        노원구 · 은평구<br><br>
        <b>문제</b><br>
        80세 이상 독거노인 비율 높음<br><br>
        <b>제안</b><br>
        · 폭염특보 시 전화·방문 안내<br>
        · 고령 독거노인 이동 지원 검토<br>
        · 복지관·경로당 연계 보호체계 강화
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("TOP 5별 개선 방향 요약")
    if not top5_priority.empty:
        st.dataframe(format_top5_table(top5_priority), use_container_width=True, hide_index=True)

    st.success("결론: 무더위쉼터 개선은 단순 증설이 아니라, 독거노인 분포와 실제 쉼터 수용능력을 함께 고려한 취약 기준별 맞춤 대응이 필요합니다.")

    st.caption("출처: 서울시 독거노인 현황 데이터, 서울시 무더위쉼터 현황 데이터, SGIS 센서스 공간정보 행정구역 경계자료 기반 직접 분석")
