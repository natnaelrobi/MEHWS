import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AEGIS-core | National Flood & Drought Command Center",
    page_icon="🛡️",
    layout="wide"
)

# Enterprise Dark Command Center Styling
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .stMetric { background-color: #111827; padding: 14px; border-radius: 8px; border: 1px solid #1f2937; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }
    h1, h2, h3, h4 { color: #f9fafb !important; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0b0f19; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #111827; border-radius: 6px; color: #9ca3af; padding: 8px 20px; border: 1px solid #1f2937; }
    .stTabs [aria-selected="true"] { background-color: #1d4ed8 !important; color: white !important; border-color: #2563eb !important; }
    div[data-testid="stSidebar"] { background-color: #030712; border-right: 1px solid #1f2937; }
    </style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000/api/v1"

# Ethiopian Regional Nodes with Flood & Drought Risk Metadata
REGIONS_DATA = [
    {"region_key": "addis_ababa", "name": "Addis Ababa City", "lat": 9.0320, "lon": 38.7421, "zone": "Central", "pcode": "ET0101", "risk_level": "Moderate", "flood_prob": 45.2, "drought_prob": 12.0, "exposed_pop": "185,400"},
    {"region_key": "dire_dawa", "name": "Dire Dawa Admin", "lat": 9.5936, "lon": 41.8661, "zone": "Eastern", "pcode": "ET1101", "risk_level": "High", "flood_prob": 78.5, "drought_prob": 35.1, "exposed_pop": "310,200"},
    {"region_key": "oromia_borena", "name": "Borena Zone (Oromia)", "lat": 4.8814, "lon": 38.0831, "zone": "Southern Oromia", "pcode": "ET0414", "risk_level": "High", "flood_prob": 22.0, "drought_prob": 89.4, "exposed_pop": "540,100"},
    {"region_key": "amhara_bahirdar", "name": "Bahir Dar & Amhara Highlands", "lat": 11.5742, "lon": 37.3614, "zone": "Northwestern Amhara", "pcode": "ET0302", "risk_level": "Critical", "flood_prob": 84.0, "drought_prob": 15.0, "exposed_pop": "420,000"},
    {"region_key": "afar_awash", "name": "Awash Valley (Afar)", "lat": 10.0000, "lon": 40.6667, "zone": "Rift Valley", "pcode": "ET0203", "risk_level": "High", "flood_prob": 82.1, "drought_prob": 64.0, "exposed_pop": "215,800"},
    {"region_key": "tigray_mekelle", "name": "Mekelle Zone (Tigray)", "lat": 13.4967, "lon": 39.4753, "zone": "Northern Tigray", "pcode": "ET0102", "risk_level": "High", "flood_prob": 61.0, "drought_prob": 45.0, "exposed_pop": "290,000"}
]
df_regions = pd.DataFrame(REGIONS_DATA)

# Top Status Metrics Bar (Flood & Drought Focus)
top_m1, top_m2, top_m3, top_m4 = st.columns(4)
top_m1.metric("AVG. FLOOD RISK", "68%", "▲ +4.2%", delta_color="inverse")
top_m2.metric("AVG. DROUGHT RISK", "42.5%", "+3.1% / 24hr")
top_m3.metric("ACTIVE SENSORS", "124", "Open-Meteo Synced")
top_m4.metric("SATELLITE COVERAGE", "98.2%", "Nominal / Online")

st.markdown("---")

# Main Layout: Sidebar navigation, Center Map, Right Priority Alerts
sidebar_col, map_col, alerts_col = st.columns([0.8, 2.2, 1.4])

with sidebar_col:
    st.markdown("### 🎛️ Navigation")
    st.markdown("📊 **Dashboard**")
    st.markdown("📍 **Risk Maps**")
    st.markdown("🚨 **Alerts**")
    st.markdown("📈 **Analytics**")
    st.markdown("---")
    st.markdown("#### Region Selector")
    selected_region_key = st.selectbox(
        "Target Zone",
        options=df_regions["region_key"].tolist(),
        format_func=lambda x: df_regions[df_regions["region_key"] == x]["name"].values[0],
        label_visibility="collapsed"
    )
    
    st.markdown("#### 🗺️ Map Layers")
    include_relief = st.checkbox("Relief Lines", value=True)
    show_heatmap = st.checkbox("Risk Heatmap", value=True)
    show_rivers = st.checkbox("River Networks", value=True)
    show_roads = st.checkbox("Road Access", value=True)
    
    st.markdown("---")
    run_ingest_btn = st.button("⚡ Run Live Ingestion", type="primary", use_container_width=True)

with map_col:
    st.markdown("### ETHIOPIA: FLOOD & DROUGHT RISK ASSESSMENT")
    
    fig_map = px.scatter_geo(
        df_regions,
        lat="lat",
        lon="lon",
        hover_name="name",
        size="flood_prob",
        color="risk_level",
        color_discrete_map={"Critical": "#ef4444", "High": "#f97316", "Moderate": "#f59e0b", "Low": "#10b981"},
        scope="africa",
        center=dict(lat=9.145, lon=40.489),
        height=450
    )
    
    fig_map.update_geos(
        fitbounds="locations",
        visible=True,
        resolution=50,
        bgcolor="#111827",
        landcolor="#1f2937",
        subunitcolor="#374151",
        countrycolor="#4b5563",
        coastlinecolor="#6b7280"
    )
    
    fig_map.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    active_reg = df_regions[df_regions["region_key"] == selected_region_key].iloc[0]
    st.markdown(f"""
        <div style="background-color: #111827; padding: 12px; border-radius: 6px; border: 1px solid #1f2937;">
            <span style="color: #60a5fa; font-weight: bold; font-size: 13px;">📍 CURRENT VIEW — LIVE NODE</span><br>
            <b style="font-size: 16px; color: white;">{active_reg['name']}</b><br>
            <small style="color: #9ca3af;">Monitoring zone telemetry. Flood Probability: {active_reg['flood_prob']}% | Drought Probability: {active_reg['drought_prob']}%</small><br>
            <div style="margin-top: 6px; font-family: monospace; font-size: 11px; color: #d1d5db;">
                COORDINATES: {active_reg['lat']}°N {active_reg['lon']}°E &nbsp;|&nbsp; SCALE: 1:250,000 &nbsp;|&nbsp; P-CODE: {active_reg['pcode']}
            </div>
        </div>
    """, unsafe_allow_html=True)

with alerts_col:
    st.markdown("### 🚨 Priority Alerts")
    st.caption("Active Model Warnings")
    
    st.markdown("""
        <div style="background-color: #111827; padding: 10px; border-radius: 6px; border-left: 4px solid #ef4444; border: 1px solid #1f2937; margin-bottom: 8px;">
            <span style="background-color: #7f1d1d; color: #fca5a5; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">CRITICAL</span>
            <span style="float: right; color: #9ca3af; font-size: 11px;">12 mins ago</span>
            <b style="display: block; color: white; margin-top: 4px; font-size: 13px;">Flash Flood Warning: Blue Nile Basin</b>
            <small style="color: #9ca3af;">Model: XGBoost Flood Detector</small>
            <div style="text-align: right;"><a href="#" style="color: #60a5fa; font-size: 11px; text-decoration: none;">Details &rarr;</a></div>
        </div>
        
        <div style="background-color: #111827; padding: 10px; border-radius: 6px; border-left: 4px solid #f97316; border: 1px solid #1f2937; margin-bottom: 8px;">
            <span style="background-color: #7c2d12; color: #fdba74; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">HIGH</span>
            <span style="float: right; color: #9ca3af; font-size: 11px;">45 mins ago</span>
            <b style="display: block; color: white; margin-top: 4px; font-size: 13px;">Drought Stress Advisory: Borena Zone</b>
            <small style="color: #9ca3af;">Model: XGBoost Drought Classifier</small>
            <div style="text-align: right;"><a href="#" style="color: #60a5fa; font-size: 11px; text-decoration: none;">Details &rarr;</a></div>
        </div>
        
        <div style="background-color: #111827; padding: 10px; border-radius: 6px; border-left: 4px solid #f59e0b; border: 1px solid #1f2937; margin-bottom: 8px;">
            <span style="background-color: #78350f; color: #fde047; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">MODERATE</span>
            <span style="float: right; color: #9ca3af; font-size: 11px;">2 hours ago</span>
            <b style="display: block; color: white; margin-top: 4px; font-size: 13px;">Awash Basin Water Level Watch</b>
            <small style="color: #9ca3af;">Model: Telemetry Pipeline</small>
            <div style="text-align: right;"><a href="#" style="color: #60a5fa; font-size: 11px; text-decoration: none;">Details &rarr;</a></div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("OPEN INCIDENT COMMAND", use_container_width=True, type="primary"):
        st.success("Incident Command Console Activated.")

st.markdown("---")

# Bottom Section: Regional Risk Indices & Precipitation Forecast
bot_col1, bot_col2 = st.columns([1, 1])

with bot_col1:
    st.markdown("### 📊 Regional Flood & Drought Indices")
    for idx, row in df_regions.iterrows():
        badge_color = "#ef4444" if row["risk_level"] in ["Critical", "High"] else "#f59e0b"
        st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <b style="color: white;">{row['name']}</b> 
                <span style="float: right; color: {badge_color}; font-weight: bold;">Flood: {row['flood_prob']}% | Drought: {row['drought_prob']}%</span>
                <div style="background-color: #1f2937; border-radius: 4px; height: 8px; width: 100%; margin-top: 4px;">
                    <div style="background-color: {badge_color}; height: 8px; border-radius: 4px; width: {row['flood_prob']}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

with bot_col2:
    st.markdown("### ☁️ Meteorological Precip. Forecast")
    st.markdown("""
        <div style="background-color: #111827; padding: 12px; border-radius: 8px; border: 1px solid #1f2937;">
            <span style="color: #60a5fa; font-weight: bold;">Heavier Rain Expected</span> <span style="float: right; background-color: #b91c1c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Watch Level 2</span><br>
            <small style="color: #9ca3af;">Est. 48hr accumulation: 120mm across Ethiopian Highlands. Convective activity peaking between 14:00 and 18:00 Local Time.</small>
        </div>
    """, unsafe_allow_html=True)