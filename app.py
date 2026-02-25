import datetime as dt
import re
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="日本周辺の地震ダッシュボード（過去1年）",
    layout="wide",
)

# Japan bounding box (roughly)
JAPAN_BBOX = {
    "minlatitude": 24.0,
    "maxlatitude": 46.0,
    "minlongitude": 122.0,
    "maxlongitude": 146.0,
}


# -----------------------------
# Helpers
# -----------------------------
def _to_date(x) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def extract_city(place: str) -> str:
    """
    USGS 'place' examples:
    - '10 km ENE of Ishinomaki, Japan' -> 'Ishinomaki'
    - 'near the east coast of Honshu, Japan' -> '不明/海域'
    """
    if place is None or (isinstance(place, float) and np.isnan(place)):
        return "不明/海域"

    s = str(place)

    # pattern: "... of <City>, <Country>"
    m = re.search(r"\bof\s+([^,]+)", s)
    if m:
        city = m.group(1).strip()
        city = re.sub(r"\s+", " ", city)
        return city

    return "不明/海域"


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_usgs_earthquakes(
    start_date: dt.date,
    end_date: dt.date,
    min_magnitude: float,
    depth_range_km: Tuple[float, float],
) -> pd.DataFrame:
    """
    Fetch earthquakes from USGS API (GeoJSON) for Japan bounding box.
    Note: USGS API expects ISO8601 dates; endtime is inclusive-ish.
    Depth filter is applied client-side (USGS API doesn't provide min/max depth parameters).
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": start_date.isoformat(),
        "endtime": end_date.isoformat(),
        "minmagnitude": float(min_magnitude),
        **JAPAN_BBOX,
        "orderby": "time",
        "limit": 20000,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    features = data.get("features", [])
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None, None])
        lon, lat, depth = (coords + [None, None, None])[:3]

        ms = props.get("time")
        t = pd.to_datetime(ms, unit="ms", utc=True) if ms is not None else pd.NaT

        rows.append(
            {
                "time_utc": t,
                "time_jst": t.tz_convert("Asia/Tokyo") if pd.notna(t) else pd.NaT,
                "magnitude": props.get("mag"),
                "place": props.get("place"),
                "depth_km": depth,
                "latitude": lat,
                "longitude": lon,
                "url": props.get("url"),
                "event_type": props.get("type"),
                "status": props.get("status"),
                "tsunami": props.get("tsunami"),
            }
        )

    df = pd.DataFrame(rows)

    # Clean types
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop invalid
    df = df.dropna(subset=["time_jst", "magnitude", "depth_km", "latitude", "longitude"])

    # Client-side depth filter
    dmin, dmax = depth_range_km
    df = df[(df["depth_km"] >= dmin) & (df["depth_km"] <= dmax)].copy()

    # Sort newest first
    df = df.sort_values("time_jst", ascending=False).reset_index(drop=True)
    return df


def kpi_block(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("条件に一致する地震データがありません。フィルタ条件を変更してください。")
        return

    total = len(df)
    max_mag = df["magnitude"].max()
    med_depth = float(df["depth_km"].median())

    idx = df["magnitude"].idxmax()
    biggest = df.loc[idx]
    biggest_text = f"M{biggest['magnitude']:.1f} / {biggest['time_jst']:%Y-%m-%d %H:%M} / {str(biggest['place'])}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("地震回数", f"{total:,}")
    c2.metric("最大マグニチュード", f"{max_mag:.1f}")
    c3.metric("中央値の深さ(km)", f"{med_depth:.1f}")
    c4.metric("最大規模（概要）", biggest_text)


def make_map(df: pd.DataFrame):
    d = df.copy()
    d["size_val"] = (d["magnitude"].clip(lower=0) ** 2)

    center = {"lat": 36.2, "lon": 138.2}

    fig = px.scatter_mapbox(
        d,
        lat="latitude",
        lon="longitude",
        color="depth_km",
        size="size_val",
        size_max=30,
        zoom=4,
        center=center,
        height=650,
        hover_name="place",
        hover_data={
            "time_jst": "|%Y-%m-%d %H:%M",
            "magnitude": ":.2f",
            "depth_km": ":.1f",
            "latitude": ":.2f",
            "longitude": ":.2f",
            "size_val": False,
            "url": True,
        },
        color_continuous_scale="Turbo",
        labels={"depth_km": "深さ(km)", "magnitude": "マグニチュード"},
        title="地震の空間分布（サイズ=規模、色=深さ）",
    )

    fig.update_traces(opacity=0.75)
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar=dict(title="深さ(km)"),
    )
    return fig


def make_timeseries(df: pd.DataFrame, granularity: str):
    if granularity == "週次":
        g = (
            df.assign(week=df["time_jst"].dt.to_period("W").dt.start_time)
            .groupby("week")
            .size()
            .reset_index(name="count")
            .rename(columns={"week": "date"})
        )
        title = "地震発生回数の推移（週次）"
        x = "date"
    else:
        g = (
            df.assign(date=df["time_jst"].dt.date)
            .groupby("date")
            .size()
            .reset_index(name="count")
        )
        title = "地震発生回数の推移（日次）"
        x = "date"

    fig = px.bar(
        g,
        x=x,
        y="count",
        title=title,
        labels={x: "日付", "count": "回数"},
        height=350,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def make_mag_hist(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="magnitude",
        nbins=30,
        title="マグニチュード分布",
        labels={"magnitude": "マグニチュード"},
        height=350,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def make_depth_mag_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="depth_km",
        y="magnitude",
        color="magnitude",
        color_continuous_scale="Plasma",
        title="深さとマグニチュードの関係",
        labels={"depth_km": "深さ(km)", "magnitude": "マグニチュード"},
        height=350,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig

def make_magnitude_wave(df: pd.DataFrame):
    d = df.sort_values("time_jst").copy()

    # rolling mean untuk efek gelombang yang lebih halus
    d["mag_ma7"] = d["magnitude"].rolling(window=7, min_periods=1).mean()

    fig = px.line(
        d,
        x="time_jst",
        y=["magnitude", "mag_ma7"],
        title="マグニチュードの推移（イベント系列 / 移動平均）",
        labels={"value": "マグニチュード", "time_jst": "時刻", "variable": "系列"},
        height=350,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def make_fft_daily_counts(df: pd.DataFrame):
    """
    FFT on daily earthquake counts (uniform daily sampling).
    Output: amplitude spectrum + top periodicities.
    """
    if df.empty:
        return px.line(title="FFT（データなし）")

    d = df.copy()
    d["date"] = d["time_jst"].dt.date

    # daily counts and fill missing days with 0 (important for FFT)
    daily = d.groupby("date").size().rename("count").to_frame()
    full_idx = pd.date_range(
        start=pd.to_datetime(daily.index.min()),
        end=pd.to_datetime(daily.index.max()),
        freq="D",
    )
    daily = daily.reindex(full_idx.date, fill_value=0)
    daily.index = pd.to_datetime(daily.index)
    y = daily["count"].to_numpy(dtype=float)

    n = len(y)
    if n < 8:
        fig = px.line(title="FFT（日次データが少なすぎます）")
        return fig

    # remove mean to reduce DC component
    y0 = y - y.mean()

    # rFFT (positive frequencies)
    fft_vals = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles per day (1/day)

    amp = np.abs(fft_vals)

    spec = pd.DataFrame({"freq_per_day": freqs, "amplitude": amp})
    spec = spec[spec["freq_per_day"] > 0].copy()  # drop zero freq
    spec["period_days"] = 1.0 / spec["freq_per_day"]

    # focus on reasonable periods (e.g., 2 to 180 days)
    spec_view = spec[(spec["period_days"] >= 2) & (spec["period_days"] <= 180)].copy()
    spec_view = spec_view.sort_values("period_days")

    fig = px.line(
        spec_view,
        x="period_days",
        y="amplitude",
        title="FFTスペクトル（日次地震回数 → 周期成分）",
        labels={"period_days": "周期（日）", "amplitude": "振幅"},
        height=350,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(type="log")  # log bikin periode pendek & panjang sama-sama kelihatan
    return fig

def add_bins_for_sankey(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      - "月別": 01月..12月
      - "季節別": 春/夏/秋/冬
    """
    d = df.copy()

    if mode == "季節別":
        # meteorological seasons (Japan common)
        def season(m: int) -> str:
            if m in (3, 4, 5):
                return "春(3-5月)"
            if m in (6, 7, 8):
                return "夏(6-8月)"
            if m in (9, 10, 11):
                return "秋(9-11月)"
            return "冬(12-2月)"

        d["time_group"] = d["time_jst"].dt.month.apply(season)
        # order for nicer display
        order = ["春(3-5月)", "夏(6-8月)", "秋(9-11月)", "冬(12-2月)"]
        d["time_group"] = pd.Categorical(d["time_group"], categories=order, ordered=True)
    else:
        d["time_group"] = d["time_jst"].dt.month.apply(lambda m: f"{m:02d}月")
        order = [f"{m:02d}月" for m in range(1, 13)]
        d["time_group"] = pd.Categorical(d["time_group"], categories=order, ordered=True)

    # depth category (km)
    d["depth_cat"] = pd.cut(
        d["depth_km"],
        bins=[-0.01, 70, 300, 10000],
        labels=["浅い(0-70km)", "中間(70-300km)", "深い(300km+)"],
    ).astype(str)

    # magnitude category
    d["mag_cat"] = pd.cut(
        d["magnitude"],
        bins=[-0.01, 3, 5, 10],
        labels=["小(0-3)", "中(3-5)", "大(5+)"],
    ).astype(str)

    return d



def make_sankey_time_depth_mag(df: pd.DataFrame, mode: str):
    """
    Sankey: time_group(月/季節) -> depth_cat -> mag_cat
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sankey図（データなし）")
        return fig

    d = add_bins_for_sankey(df, mode=mode).dropna(subset=["time_group", "depth_cat", "mag_cat"]).copy()

    # links: time -> depth, depth -> mag
    a = d.groupby(["time_group", "depth_cat"]).size().reset_index(name="value")
    b = d.groupby(["depth_cat", "mag_cat"]).size().reset_index(name="value")

    # nodes
    nodes = (
        pd.Index(a["time_group"].astype(str))
        .append(pd.Index(a["depth_cat"].astype(str)))
        .append(pd.Index(b["mag_cat"].astype(str)))
        .unique()
        .tolist()
    )
    node_index = {label: i for i, label in enumerate(nodes)}

    sources, targets, values, link_colors = [], [], [], []

    def _rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # time -> depth
    for _, row in a.iterrows():
        sources.append(node_index[str(row["time_group"])])
        targets.append(node_index[str(row["depth_cat"])])
        values.append(int(row["value"]))
        link_colors.append(_rgba("#033A62", 0.35))

    # depth -> mag
    for _, row in b.iterrows():
        sources.append(node_index[str(row["depth_cat"])])
        targets.append(node_index[str(row["mag_cat"])])
        values.append(int(row["value"]))
        link_colors.append(_rgba("#8f0a0a", 0.40))

    title = f"Sankey図：{mode} → 深さカテゴリ → 規模カテゴリ（件数）"

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=14,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0.15)", width=1),
                    label=nodes,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=540,
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig

def show_auto_insights(df: pd.DataFrame, sankey_mode: str) -> None:
    if df.empty:
        st.info("インサイトを算出するデータがありません。")
        return

    d = df.copy()
    d["month"] = d["time_jst"].dt.month
    d["month_label"] = d["month"].apply(lambda m: f"{m:02d}月")

    def season(m: int) -> str:
        if m in (3, 4, 5):
            return "春(3-5月)"
        if m in (6, 7, 8):
            return "夏(6-8月)"
        if m in (9, 10, 11):
            return "秋(9-11月)"
        return "冬(12-2月)"

    d["season_label"] = d["month"].apply(season)

    # categories (same as Sankey)
    d["depth_cat"] = pd.cut(
        d["depth_km"],
        bins=[-0.01, 70, 300, 10000],
        labels=["浅い(0-70km)", "中間(70-300km)", "深い(300km+)"],
    ).astype(str)

    d["mag_cat"] = pd.cut(
        d["magnitude"],
        bins=[-0.01, 3, 5, 10],
        labels=["小(0-3)", "中(3-5)", "大(5+)"],
    ).astype(str)

    total = len(d)

    # peak time group
    if sankey_mode == "季節別":
        peak_series = d["season_label"].value_counts()
        peak_label = peak_series.index[0]
        peak_count = int(peak_series.iloc[0])
        time_title = "最多の季節"
    else:
        peak_series = d["month_label"].value_counts()
        peak_label = peak_series.index[0]
        peak_count = int(peak_series.iloc[0])
        time_title = "最多の月"

    # shallow proportion
    shallow_count = int((d["depth_km"] < 70).sum())
    shallow_pct = shallow_count / total * 100

    # M5+ proportion
    m5_count = int((d["magnitude"] >= 5.0).sum())
    m5_pct = m5_count / total * 100

    # dominant Sankey path (time -> depth -> mag)
    if sankey_mode == "季節別":
        tg = "season_label"
    else:
        tg = "month_label"

    top_path = (
        d.groupby([tg, "depth_cat", "mag_cat"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(1)
    )
    if len(top_path) == 1:
        tp = top_path.iloc[0]
        top_path_text = f"{tp[tg]} → {tp['depth_cat']} → {tp['mag_cat']}（{int(tp['count']):,}件）"
    else:
        top_path_text = "—"

    st.subheader("自動インサイト（要約）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(time_title, f"{peak_label}", f"{peak_count:,}件")
    c2.metric("浅い地震の割合(0-70km)", f"{shallow_pct:.1f}%", f"{shallow_count:,}件")
    c3.metric("M5以上の割合", f"{m5_pct:.2f}%", f"{m5_count:,}件")
    c4.metric("最頻パス（Sankey）", top_path_text)

    st.markdown(
        f"""
- **{time_title}**は **{peak_label}**（{peak_count:,}件）。
- **浅い地震(0–70km)** は **{shallow_count:,}件（{shallow_pct:.1f}%）**。
- **M5以上** は **{m5_count:,}件（{m5_pct:.2f}%）**。
- Sankey上で最も多い流れは **{top_path_text}**。
"""
    )
# -----------------------------
# UI
# -----------------------------
st.title("日本周辺の地震データ可視化ダッシュボード（過去1年）")
st.caption("USGS Earthquake APIを使用。日本周辺（緯度24–46、経度122–146）の地震を対象に可視化します。")
st.caption("注意：USGSデータは速報値を含むため、後日更新される場合があります。")

today = dt.date.today()
default_start = today - dt.timedelta(days=365)

with st.sidebar:
    st.header("フィルタ")
    date_range = st.date_input(
        "期間",
        value=(default_start, today),
        min_value=today - dt.timedelta(days=365 * 5),
        max_value=today,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, today

    min_mag = st.slider("最小マグニチュード", min_value=0.0, max_value=7.0, value=2.5, step=0.1)
    depth_min, depth_max = st.slider("深さ(km)", min_value=0.0, max_value=700.0, value=(0.0, 300.0), step=10.0)
    granularity = st.radio("集計単位", options=["日次", "週次"], horizontal=True)
    st.divider()
    st.subheader("Sankey設定")
    sankey_mode = st.radio("表示粒度", options=["月別", "季節別"], horizontal=True)
    st.divider()
    st.write("対象範囲（固定）")
    st.write(f"緯度: {JAPAN_BBOX['minlatitude']}〜{JAPAN_BBOX['maxlatitude']}")
    st.write(f"経度: {JAPAN_BBOX['minlongitude']}〜{JAPAN_BBOX['maxlongitude']}")

with st.spinner("USGSからデータを取得中..."):
    df = fetch_usgs_earthquakes(
        start_date=_to_date(start_date),
        end_date=_to_date(end_date),
        min_magnitude=float(min_mag),
        depth_range_km=(float(depth_min), float(depth_max)),
    )

kpi_block(df)

if df.empty:
    st.stop()

left, right = st.columns([2.2, 1.0], gap="large")
with left:
    st.plotly_chart(make_map(df), use_container_width=True)
    st.subheader("季節性の可視化（Sankey図）")
    st.plotly_chart(make_sankey_time_depth_mag(df, mode=sankey_mode), use_container_width=True)
    st.caption("注：本Sankey図は「月/季節→深さ→規模」の分布を件数で示すものであり、因果関係を直接示すものではありません。")
    show_auto_insights(df, sankey_mode)
with right:
    st.plotly_chart(make_timeseries(df, granularity), use_container_width=True)
    st.plotly_chart(make_mag_hist(df), use_container_width=True)
    st.plotly_chart(make_depth_mag_scatter(df), use_container_width=True)
    st.plotly_chart(make_magnitude_wave(df), use_container_width=True)
    st.plotly_chart(make_fft_daily_counts(df), use_container_width=True)

st.subheader("データテーブル（上位200件）")
show_cols = ["time_jst", "magnitude", "depth_km", "place", "latitude", "longitude", "url"]
st.dataframe(df[show_cols].head(200), use_container_width=True, hide_index=True)

csv = df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "フィルタ後データをCSVでダウンロード",
    data=csv,
    file_name="japan_earthquakes_filtered.csv",
    mime="text/csv",
)

# st.subheader("簡単な考察（例）")
# st.markdown(
#     """
# - 地震はプレート境界付近に集中しやすく、特定の海域周辺で点群が密になる傾向が見られる。
# - 最小マグニチュードや深さ条件を変えることで、観測される地震活動の見え方が大きく変化する。
# - 深さとマグニチュードの関係は単純ではなく、複数の要因（発生域・観測条件など）が影響する可能性がある。
# """
# )
