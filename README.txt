# Japan Earthquake Dashboard (Past 1 Year) — Streamlit + USGS

A Streamlit dashboard that fetches earthquake events from the **USGS Earthquake API** and visualizes earthquakes around Japan (fixed bounding box: **lat 24–46, lon 122–146**) for an interactive, filterable analysis.

---

## English

### Overview
This project provides an interactive dashboard to explore earthquakes around Japan using the USGS FDSN Event API (GeoJSON).  
You can filter by date range, minimum magnitude, and depth range, then inspect patterns via maps, charts, Sankey diagrams, and frequency analysis (FFT).

### Data Source
- **USGS Earthquake API (FDSN Event Web Service)**  
  https://earthquake.usgs.gov/fdsnws/event/1/

> Note: USGS data may include preliminary reports and can be updated later.

### Features
- **Interactive Map (Plotly Mapbox)**  
  - Marker size = magnitude (scaled)
  - Marker color = depth (km)
- **KPI Summary**
  - Total events
  - Maximum magnitude
  - Median depth
  - Largest event overview (time/place)
- **Time Series**
  - Daily or weekly event counts
- **Distributions & Relationships**
  - Magnitude histogram
  - Depth vs magnitude scatter
  - Magnitude “wave” line chart + rolling mean
- **Frequency Analysis**
  - FFT on daily earthquake counts to highlight periodic components
- **Seasonality Sankey**
  - Time group (by **month** or **season**) → depth category → magnitude category
- **Table + CSV Export**
  - Shows top 200 rows
  - Download filtered results as CSV

### Fixed Target Region (Bounding Box)
The dashboard always uses the following region for Japan (rough):
- Latitude: **24.0 – 46.0**
- Longitude: **122.0 – 146.0**

### Requirements
- Python 3.9+ recommended
- Dependencies are listed in `requirements.txt`

### Setup
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

### Output Columns (Main Fields)
The fetched dataset includes:
- `time_utc`, `time_jst`
- `magnitude`
- `depth_km`
- `place`
- `latitude`, `longitude`
- `url`
- `event_type`, `status`, `tsunami`

### Notes / Limitations
- The USGS API query is limited to **20,000** events (`limit=20000`).
- Depth filtering is applied **client-side** after fetching, because the API does not provide min/max depth parameters directly.
- The Sankey diagram shows **counts/distributions** (not causality).

---

## 日本語

### 概要
このプロジェクトは、**USGS Earthquake API**（FDSN Event Web Service）から日本周辺の地震データ（GeoJSON）を取得し、Streamlitで可視化するダッシュボードです。  
期間・最小マグニチュード・深さ範囲をフィルタし、地図や各種グラフ、Sankey図、FFT解析などで傾向を確認できます。

### データソース
- **USGS Earthquake API（FDSN Event Web Service）**  
  https://earthquake.usgs.gov/fdsnws/event/1/

> 注意：USGSの地震データは速報値を含み、後日更新される場合があります。

### 主な機能
- **地図（Plotly Mapbox）**
  - 点のサイズ = 規模（マグニチュード）
  - 点の色 = 深さ（km）
- **KPI表示**
  - 地震回数、最大マグニチュード、深さ中央値、最大規模イベント概要
- **時系列**
  - 日次 / 週次の発生回数
- **分布・関係性**
  - マグニチュード分布、深さ×マグニチュード散布図
  - マグニチュード推移（イベント系列＋移動平均）
- **周波数解析**
  - 日次発生回数に対するFFT（周期成分の可視化）
- **季節性の可視化（Sankey図）**
  - 月別 or 季節別 → 深さカテゴリ → 規模カテゴリ（件数）
- **データテーブル & CSVダウンロード**
  - 上位200件表示、フィルタ後データをCSV出力

### 対象範囲（固定）
日本周辺（概略）のバウンディングボックスを固定で使用します：
- 緯度：**24.0 – 46.0**
- 経度：**122.0 – 146.0**

### 必要環境
- Python 3.9+ 推奨
- 依存関係は `requirements.txt` に記載

### セットアップ
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 実行
```bash
streamlit run app.py
```

### 主なカラム
- `time_utc`, `time_jst`
- `magnitude`
- `depth_km`
- `place`
- `latitude`, `longitude`
- `url`
- `event_type`, `status`, `tsunami`

### 注意点 / 制限
- API取得件数は最大 **20,000件**（`limit=20000`）。
- 深さ条件はAPI側では直接指定できないため、取得後に**クライアント側でフィルタ**しています。
- Sankey図は分布（件数）を示すもので、因果関係を直接示すものではありません。

---

## License
TBD (add your preferred license, e.g., MIT)
