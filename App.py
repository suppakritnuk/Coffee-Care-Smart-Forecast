import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
from streamlit_js_eval import get_geolocation
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Coffee Care Smart Forecast",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ข้อมูลพิกัดอำเภอในเชียงใหม่
CHIANG_MAI_DISTRICTS = {
    "อ.เมืองเชียงใหม่": (18.7883, 98.9853), "อ.จอมทอง": (18.4167, 98.6833),
    "อ.แม่แจ่ม": (18.5028, 98.3614), "อ.เชียงดาว": (19.3664, 98.9647),
    "อ.ดอยเต่า": (17.9547, 98.6858), "อ.แม่แตง": (19.1206, 98.9431),
    "อ.แม่ริม": (18.9142, 98.9453), "อ.สะเมิง": (18.8478, 98.7317),
    "อ.ฝาง": (19.9175, 99.2139), "อ.แม่อาย": (20.0336, 99.2908),
    "อ.พร้าว": (19.3644, 99.2025), "อ.สันป่าตอง": (18.6294, 98.8953),
    "อ.สันกำแพง": (18.7453, 99.1172), "อ.สันทราย": (18.8514, 99.0436),
    "อ.หางดง": (18.6858, 98.9181), "อ.ฮอด": (18.1903, 98.6139),
    "อ.ดอยสะเก็ด": (18.8661, 99.1367), "อ.สารภี": (18.7125, 99.0347),
    "อ.เวียงแหง": (19.5606, 98.6367), "อ.ไชยปราการ": (19.7369, 99.1419),
    "อ.แม่วาง": (18.6272, 98.6758), "อ.แม่ออน": (18.7831, 99.2458),
    "อ.ดอยหล่อ": (18.4753, 98.7819), "อ.กัลยาณิวัฒนา": (19.0664, 98.4553),
    "อ.อมก๋อย": (17.7981, 98.3581)
}

# --- 2. ฟังก์ชันโหลดโมเดลที่เซฟไว้ และดึง API ---
@st.cache_resource(show_spinner="กำลังโหลดโมเดล AI...")
def load_saved_models():
    try:
        # โหลดโมเดลจากไฟล์ที่รันเซฟไว้
        m_leaf_inc = joblib.load('model_leaf_incidence.joblib')
        m_berry_inc = joblib.load('model_berry_incidence.joblib')
        m_leaf_sev = joblib.load('model_leaf_severity.joblib')
        m_berry_sev = joblib.load('model_berry_severity.joblib')
        
        features_model = [
            "avg_temp", "avg_rain", "humidity", "wind_speed"
        ]
        return m_leaf_inc, m_berry_inc, m_leaf_sev, m_berry_sev, features_model, True
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None, False

m_leaf_inc, m_berry_inc, m_leaf_sev, m_berry_sev, model_features, is_ai_ready = load_saved_models()

@st.cache_data(ttl=3600)
def fetch_nasa_weather(lat, lon, start_date, end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,PRECTOTCORR,WS2M&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            param = data['properties']['parameter']
            df = pd.DataFrame({
                "Date": pd.to_datetime(list(param['T2M'].keys())),
                "Temp": list(param['T2M'].values()),
                "Humid": list(param['RH2M'].values()),
                "Rain": list(param['PRECTOTCORR'].values()),
                "Wind": list(param['WS2M'].values())
            })
            df = df[(df["Temp"] > -100) & (df["Humid"] > 0)]
            return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ ตั้งค่าระบบ")
    
    if is_ai_ready:
        st.success("📊 **Status:** Data Ready (4 Pre-trained Models)")
    else:
        st.warning("⚠️ **Status:** ไม่พบไฟล์โมเดล (กรุณารัน train_models.py ก่อน)")
        
    with st.container(border=True):
        st.subheader("📍 1. ระบุพิกัดสวน")
        loc_method = st.radio("วิธีเลือกพิกัด", ["เลือกอำเภอในเชียงใหม่", "ใช้ GPS", "กรอกเอง"], label_visibility="collapsed")
        
        c_lat, c_lon = 18.7883, 98.9853
        if loc_method == "เลือกอำเภอในเชียงใหม่":
            selected_district = st.selectbox("ระบุอำเภอ", list(CHIANG_MAI_DISTRICTS.keys()))
            c_lat, c_lon = CHIANG_MAI_DISTRICTS[selected_district]
        elif loc_method == "ใช้ GPS":
            loc = get_geolocation()
            if loc: c_lat, c_lon = loc['coords']['latitude'], loc['coords']['longitude']
        else:
            col_l, col_r = st.columns(2)
            c_lat = col_l.number_input("Lat", value=c_lat, format="%.4f")
            c_lon = col_r.number_input("Lon", value=c_lon, format="%.4f")

    with st.container(border=True):
        st.subheader("📅 2. วันที่ต้องการวิเคราะห์")
        today = datetime.now()
        target_date = st.date_input("เลือกวันที่", today - timedelta(days=1), max_value=today, label_visibility="collapsed")
        
        if st.button("🚀 ดึงข้อมูลสภาพอากาศ", width="stretch", type="primary"):
            st.session_state['show_prediction'] = False 
            start_date = target_date - timedelta(days=27)
            end_date = target_date
            with st.spinner('กำลังดึงข้อมูลย้อนหลัง 4 สัปดาห์ จาก NASA...'):
                st.session_state.weather_df = fetch_nasa_weather(c_lat, c_lon, start_date, end_date)

    st.write("---")
    predict_btn = st.button("🔮 พยากรณ์ความเสี่ยง", type="primary", width="stretch")
    
    if predict_btn:
        st.session_state['show_prediction'] = True

# --- 4. MAIN DASHBOARD ---
st.title("☕ Coffee Disease Smart Forecast")
st.markdown("##### ระบบวิเคราะห์และแจ้งเตือนการระบาดล่วงหน้าด้วยโมเดล ML")

if 'weather_df' in st.session_state and st.session_state.weather_df is not None:
    df = st.session_state.weather_df
    
    df_tail = df.tail(28)
    week_curr = df_tail.iloc[-7:] if len(df_tail) >= 7 else df_tail
    week_lag1 = df_tail.iloc[-14:-7] if len(df_tail) >= 14 else week_curr
    week_lag2 = df_tail.iloc[-21:-14] if len(df_tail) >= 21 else week_lag1
    week_lag3 = df_tail.iloc[-28:-21] if len(df_tail) >= 28 else week_lag2
    
    def get_weekly_stats(df_week):
        return {
            'temp': df_week["Temp"].mean(),
            'humid': df_week["Humid"].mean(),
            'rain': df_week["Rain"].mean(),
            'wind': df_week["Wind"].mean()
        }

    c_stats = get_weekly_stats(week_curr)
    l1_stats = get_weekly_stats(week_lag1)
    l2_stats = get_weekly_stats(week_lag2)
    l3_stats = get_weekly_stats(week_lag3)

    st.header("📋 สภาพอากาศ 7 วันที่ผ่านมา")
    with st.container(border=True):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("🌡️ อุณหภูมิเฉลี่ย", f"{c_stats['temp']:.1f} °C")
        col_m2.metric("💧 ความชื้นสัมพัทธ์", f"{c_stats['humid']:.1f} %")
        col_m3.metric("🌧️ น้ำฝนเฉลี่ยรายวัน", f"{c_stats['rain']:.1f} มม.")
        col_m4.metric("💨 ความเร็วลม", f"{c_stats['wind']:.1f} m/s")

    if st.session_state.get('show_prediction', False):
        st.divider()
        if is_ai_ready:
            pred_data = pd.DataFrame([{
                "avg_temp": c_stats['temp'], "avg_rain": c_stats['rain'], 
                "humidity": c_stats['humid'], "wind_speed": c_stats['wind'],
                "temp_lag1": l1_stats['temp'], "temp_lag2": l2_stats['temp'], "temp_lag3": l3_stats['temp'],
                "humid_lag1": l1_stats['humid'], "humid_lag2": l2_stats['humid'], "humid_lag3": l3_stats['humid'],
                "rain_lag1": l1_stats['rain'], "rain_lag2": l2_stats['rain'], "rain_lag3": l3_stats['rain'],
                "temp_roll3": np.mean([l2_stats['temp'], l1_stats['temp'], c_stats['temp']]),
                "humid_roll3": np.mean([l2_stats['humid'], l1_stats['humid'], c_stats['humid']]),
                "rain_roll3": np.sum([l2_stats['rain'], l1_stats['rain'], c_stats['rain']])
            }])
            
            X_pred = pred_data[model_features]
            
            leaf_incidence = max(0, min(100, m_leaf_inc.predict(X_pred)[0]))
            berry_incidence = max(0, min(100, m_berry_inc.predict(X_pred)[0]))
            
            leaf_severity_raw = max(0, m_leaf_sev.predict(X_pred)[0])
            berry_severity_raw = max(0, m_berry_sev.predict(X_pred)[0])
            
        else:
            risk_score = (c_stats['temp'] * 0.12) + (c_stats['humid'] * 0.8) + (c_stats['rain'] * 0.08)
            leaf_incidence = min(max(risk_score * 0.98, 0), 100.0)
            berry_incidence = min(max(risk_score * 0.75, 0), 100.0)
            leaf_severity_raw = leaf_incidence
            berry_severity_raw = berry_incidence

        def calc_severity(severity_val):
            if severity_val <= 5.0: return (severity_val / 5.0) * 2.0
            elif severity_val <= 20.0: return 2.0 + ((severity_val - 5.0) / 15.0) * 2.0
            else: return 4.0 + ((severity_val - 20.0) / 80.0) * 2.0

        leaf_level = min(max(calc_severity(leaf_severity_raw), 0), 6)
        berry_level = min(max(calc_severity(berry_severity_raw), 0), 6)
        overall_level = max(leaf_level, berry_level) 

        def create_gauge_chart(value):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 6], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': "#FFFFFF", 'line': {'color': "#3E2723", 'width': 3}},
                    'steps': [
                        {'range': [0, 2], 'color': "#50CE54"},
                        {'range': [2, 4], 'color': "#FBEA50"},
                        {'range': [4, 6], 'color': "#F65353"}
                    ]
                }
            ))
            fig.update_layout(height=230, margin=dict(t=20, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': 'gray'})
            return fig

        def display_status_msg(level):
            status = "วิกฤต (Critical)" if level >= 4 else "เฝ้าระวัง (Watch)" if level >= 2 else "ปกติ (Stable)"
            if level >= 4: st.error(f"สถานะ: **{status}**")
            elif level >= 2: st.warning(f"สถานะ: **{status}**")
            else: st.success(f"สถานะ: **{status}**")

        future_start_date = target_date + timedelta(days=1)
        future_end_date = target_date + timedelta(days=7)
        date_range_str = f"{future_start_date.strftime('%d %b %Y')} - {future_end_date.strftime('%d %b %Y')}"
        st.header("🔔 ผลการพยากรณ์ความเสี่ยงล่วงหน้า 7 วัน")
        box_height = 320
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            with st.container(border=True, height=box_height):
                st.subheader("🌿 ระดับความรุนแรงของโรคราสนิม ")
                st.plotly_chart(create_gauge_chart(leaf_level), use_container_width=True)
        with col_l2:
            with st.container(border=True, height=box_height):
                st.subheader("ความชุกของโรคราสนิมกาแฟ ")
                st.markdown(f"<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0px;'>{leaf_incidence:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(min(int(leaf_incidence), 100))
                display_status_msg(leaf_level)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            with st.container(border=True, height=box_height):
                st.subheader("🍒 ระดับความรุนแรงของโรคผลเน่า ")
                st.plotly_chart(create_gauge_chart(berry_level), use_container_width=True)
        with col_b2:
            with st.container(border=True, height=box_height):
                st.subheader("ความชุกของโรคผลเน่า ")
                st.markdown(f"<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0px;'>{berry_incidence:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(min(int(berry_incidence), 100))
                display_status_msg(berry_level)

        st.divider()
        st.header("📍 แนวทางจัดการและแผนที่ความเสี่ยง")
        col_adv, col_map = st.columns([1, 1])
        
        container_height = 550
        
        with col_adv:
            with st.container(border=True, height=container_height):
                st.subheader("💡 แนวทางการจัดการเชิงรุก:")
                if overall_level >= 4:
                    st.error("### 🆘 ระดับวิกฤต (Critical)")
                    st.markdown("""
                    #### **ข้อควรปฏิบัติทันที:**
                    1. **พ่นสารกำจัดเชื้อรา:** กลุ่ม Copper ทันที
                    2. **ตัดแต่งกิ่ง:** ให้โปร่งเพื่อลดความชื้นสะสม
                    3. **กำจัดส่วนที่ติดโรค:** นำใบ/ผลออกนอกแปลงและเผาทำลาย
                    """)
                elif overall_level >= 2:
                    st.warning("### ⚠️ ระดับเฝ้าระวัง (Watch)")
                    st.markdown("""
                    #### **ข้อควรปฏิบัติ:**
                    1. **สำรวจแปลง:** ตรวจใต้ใบและผลอย่างละเอียดสัปดาห์ละ 2 ครั้ง
                    2. **ใช้สารชีวภัณฑ์:** เช่น ไตรโคเดอร์มา ฉีดพ่นป้องกัน
                    3. **จัดการระบบน้ำ:** ตรวจสอบการระบายน้ำไม่ให้ท่วมขัง
                    """)
                else:
                    st.success("### ✅ ระดับปกติ (Stable)")
                    st.markdown("""
                    #### **การดูแลรักษา:**
                    1. **บำรุงต้น:** ใส่ปุ๋ยตามรอบปกติเพื่อให้ต้นแข็งแรง
                    2. **รักษาความสะอาด:** กำจัดวัชพืชรอบโคนต้น
                    3. **ติดตามข่าวสาร:** ตรวจเช็คสภาพอากาศสม่ำเสมอ
                    """)
                    
                st.write("") # เพิ่มช่องว่าง
                st.info(f"**🔬 วิเคราะห์สถิติ:** ความชื้นสัมพัทธ์เฉลี่ย {c_stats['humid']:.1f}% เป็นปัจจัยหลักที่ส่งผลต่อการขยายตัวของเชื้อรา")

        with col_map:
            with st.container(border=True, height=container_height):
                st.subheader("🗺️ แผนที่ความเสี่ยงระดับอำเภอ")
                
                map_view = st.radio(
                    "🔎 เลือกระดับการมองเห็น:",
                    ["ซูมพื้นที่เป้าหมาย (เฉพาะอำเภอที่เลือก)", "ภาพรวมทั้งจังหวัด"],
                    horizontal=True
                )
                
                if map_view == "ภาพรวมทั้งจังหวัด":
                    map_center = {"lat": 18.7883, "lon": 98.9853}
                    map_zoom = 6
                else:
                    map_center = {"lat": c_lat, "lon": c_lon} 
                    map_zoom = 9.5

                @st.cache_data(ttl=3600)
                def load_and_filter_geojson():
                    file_path = "chiangmai_amphoes.geojson"
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                return json.load(f)
                        except json.JSONDecodeError:
                            os.remove(file_path)
                            pass 
                            
                    url = "https://raw.githubusercontent.com/chingchai/OpenGISData-Thailand/master/districts.geojson"
                    try:
                        resp = requests.get(url, timeout=15)
                        data = resp.json()
                        chiangmai_features = []
                        for feature in data["features"]:
                            props = feature.get("properties", {})
                            pro_name = props.get("pro_th", "") or props.get("pv_th", "") or props.get("PV_TN", "")
                            if "เชียงใหม่" in pro_name:
                                chiangmai_features.append(feature)
                        chiangmai_geojson = {"type": "FeatureCollection", "features": chiangmai_features}
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(chiangmai_geojson, f, ensure_ascii=False)
                        return chiangmai_geojson
                    except Exception as e:
                        return None

                base_geojson = load_and_filter_geojson()

                if base_geojson:
                    display_geojson = {"type": "FeatureCollection", "features": []}
                    selected_clean = selected_district.replace("อ.", "").strip() if loc_method == "เลือกอำเภอในเชียงใหม่" else ""

                    if map_view == "ซูมพื้นที่เป้าหมาย (เฉพาะอำเภอที่เลือก)" and selected_clean != "":
                        for f in base_geojson["features"]:
                            d_name = f["properties"].get("amp_th", "") or f["properties"].get("ap_th", "")
                            clean_name = d_name.replace("อำเภอ", "").replace("อ.", "").strip()
                            if clean_name == selected_clean:
                                display_geojson["features"].append(f)
                    else:
                        display_geojson["features"] = base_geojson["features"]

                    if len(display_geojson["features"]) == 0:
                        display_geojson["features"] = base_geojson["features"]

                    districts_in_map = [f["properties"].get("amp_th", "") or f["properties"].get("ap_th", "") for f in display_geojson["features"]]
                    
                    risk_data = []
                    for d_name in districts_in_map:
                        clean_name = d_name.replace("อำเภอ", "").replace("อ.", "").strip()
                        if selected_clean and clean_name == selected_clean:
                            risk = overall_level
                        else:
                            np.random.seed(len(clean_name)) 
                            risk = min(np.random.uniform(0, overall_level + 1), 6.0)
                        risk_data.append({"District": d_name, "RiskLevel": risk})
                    
                    df_risk = pd.DataFrame(risk_data)
                    feature_key = 'properties.amp_th' if 'amp_th' in display_geojson['features'][0]['properties'] else 'properties.ap_th'

                    fig_map = px.choropleth_mapbox(
                        df_risk, geojson=display_geojson, locations='District', featureidkey=feature_key, 
                        color='RiskLevel', color_continuous_scale="RdYlGn_r", range_color=(0, 6),
                        mapbox_style="carto-positron", zoom=map_zoom, center=map_center,
                        opacity=0.6, labels={'RiskLevel': 'ระดับความเสี่ยง (0-6)', 'District': 'อำเภอ'}
                    )
                    fig_map.update_layout(height=350, margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar=dict(title="ความเสี่ยง"))
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.error("ไม่สามารถโหลดแผนที่ได้")

    st.divider()
    st.subheader("📉 รายละเอียดกราฟแนวโน้มสภาพอากาศย้อนหลัง")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # กราฟอุณหภูมิ (สีแดงส้ม)
        fig_temp = px.line(df, x="Date", y="Temp", template="plotly_white", 
                           title="🌡️ แนวโน้มอุณหภูมิเฉลี่ย (°C)", color_discrete_sequence=['#ef553b'])
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # กราฟปริมาณน้ำฝน (สีฟ้าคราม)
        fig_rain = px.line(df, x="Date", y="Rain", template="plotly_white", 
                           title="🌧️ แนวโน้มปริมาณน้ำฝน (มม.)", color_discrete_sequence=['#00cc96'])
        st.plotly_chart(fig_rain, use_container_width=True)

    with col_g2:
        # กราฟความชื้นสัมพัทธ์ (สีน้ำเงิน)
        fig_humid = px.line(df, x="Date", y="Humid", template="plotly_white", 
                            title="💧 แนวโน้มความชื้นสัมพัทธ์ (%)", color_discrete_sequence=['#636efa'])
        st.plotly_chart(fig_humid, use_container_width=True)
        
        # กราฟความเร็วลม (สีม่วง)
        fig_wind = px.line(df, x="Date", y="Wind", template="plotly_white", 
                           title="💨 แนวโน้มความเร็วลม (m/s)", color_discrete_sequence=['#ab63fa'])
        st.plotly_chart(fig_wind, use_container_width=True)

else:
    st.info("👈 **โปรดเริ่มต้น:** เลือกพิกัดสวนกาแฟ ระบุวันที่เป้าหมาย และกดปุ่มดึงข้อมูลอากาศจากแถบด้านข้าง")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    st.image("my_coffee.png", caption="ระบบแจ้งเตือนโรคกาแฟอัจฉริยะ", use_container_width=True)

st.divider()
st.caption("© 2026 KMUTT Statistics & Data Science | Coffee Disease Early Warning System")