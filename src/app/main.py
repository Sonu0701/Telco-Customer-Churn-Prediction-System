"""
STREAMLIT UI - Telco Customer Churn Prediction Dashboard
=========================================================

Production-ready Streamlit frontend for the FastAPI + MLflow + Docker + Render stack.

Run locally:
    streamlit run streamlit_app.py

Environment variables:
    API_URL  - Base URL of the FastAPI backend (default: http://localhost:8000)
"""

import streamlit as st
import requests
import os
import json
import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Backend URL ──────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0b0f1a;
    color: #e2e8f0;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }

/* ── Brand Header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2rem;
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 60%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1;
    margin: 0;
}
.brand-sub {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #38bdf8;
    border-left: 2px solid #38bdf8;
    padding-left: 10px;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* ── Card ── */
.card {
    background: #131929;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Prediction Banner ── */
.pred-churn {
    background: linear-gradient(135deg, #450a0a 0%, #1a0a1a 100%);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.pred-safe {
    background: linear-gradient(135deg, #042f1e 0%, #0a1a12 100%);
    border: 1px solid #22c55e;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.pred-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.pred-icon { font-size: 3rem; }
.pred-desc { font-size: 0.78rem; color: #94a3b8; margin-top: 0.5rem; }

/* ── Risk Meter ── */
.risk-bar-wrap {
    background: #1e2d45;
    border-radius: 100px;
    height: 10px;
    margin: 1rem 0;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}

/* ── Status Pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0f1e30;
    border: 1px solid #1e3a5f;
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 0.7rem;
    color: #7dd3fc;
    letter-spacing: 1px;
}
.dot { width:7px; height:7px; border-radius:50%; background:#22c55e;
       display:inline-block; box-shadow: 0 0 6px #22c55e; }

/* ── Metric Card ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-box {
    flex: 1;
    background: #131929;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-lbl { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 2px; }

/* ── Streamlit Widget Overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}
div[data-testid="stSidebar"] {
    background: #0b0f1a !important;
    border-right: 1px solid #1e2d45;
}

/* ── Divider ── */
.subtle-divider {
    border: none;
    border-top: 1px solid #1e2d45;
    margin: 1.5rem 0;
}

/* ── JSON Block ── */
.json-block {
    background: #060b14;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.72rem;
    color: #7dd3fc;
    overflow-x: auto;
    white-space: pre;
}

/* ── Tip Box ── */
.tip-box {
    background: #0a1628;
    border-left: 3px solid #818cf8;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.73rem;
    color: #94a3b8;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper: Health Check ──────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


# ─── Helper: Predict ──────────────────────────────────────────────────────────
def call_predict(payload: dict):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, str(e)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header" style="margin-bottom:1.5rem">
        <div>
            <div class="brand-title">ChurnSight</div>
            <div class="brand-sub">Telco Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    api_ok = check_api_health()
    status_dot = "🟢" if api_ok else "🔴"
    status_text = "API Online" if api_ok else "API Offline"
    st.markdown(f"""
    <div class="status-pill">
        <span class="dot" style="background:{'#22c55e' if api_ok else '#ef4444'};
              box-shadow:0 0 6px {'#22c55e' if api_ok else '#ef4444'}"></span>
        {status_text}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Backend Config</div>', unsafe_allow_html=True)

    custom_url = st.text_input("API Base URL", value=API_URL, key="api_url_input",
                                help="Set API_URL env variable or override here.")
    if custom_url != API_URL:
        API_URL = custom_url

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Stack Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#64748b; line-height:2">
    🔷 <b style="color:#94a3b8">FastAPI</b> — REST backend<br>
    🟣 <b style="color:#94a3b8">MLflow</b> — Model registry<br>
    🐳 <b style="color:#94a3b8">Docker</b> — Containerised<br>
    🚀 <b style="color:#94a3b8">Render</b> — Cloud deploy<br>
    🤖 <b style="color:#94a3b8">XGBoost</b> — Churn model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Quick Fill</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    preset_high = col_a.button("⚠️ High Risk", use_container_width=True)
    preset_low  = col_b.button("✅ Low Risk",  use_container_width=True)

    st.markdown("""
    <div class="tip-box">
    💡 Month-to-month contracts + Fiber optic + Electronic check = highest churn risk profile.
    </div>
    """, unsafe_allow_html=True)


# ─── Presets ──────────────────────────────────────────────────────────────────
HIGH_RISK = dict(
    gender="Female", Partner="No", Dependents="No",
    PhoneService="Yes", MultipleLines="No",
    InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
    DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
    Contract="Month-to-month", PaperlessBilling="Yes",
    PaymentMethod="Electronic check", tenure=1,
    MonthlyCharges=85.0, TotalCharges=85.0
)
LOW_RISK = dict(
    gender="Male", Partner="Yes", Dependents="Yes",
    PhoneService="Yes", MultipleLines="Yes",
    InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
    DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No", StreamingMovies="No",
    Contract="Two year", PaperlessBilling="No",
    PaymentMethod="Credit card (automatic)", tenure=60,
    MonthlyCharges=45.0, TotalCharges=2700.0
)

defaults = HIGH_RISK if preset_high else (LOW_RISK if preset_low else None)


def dv(key, fallback):
    """Return preset value if loaded, else session state or fallback."""
    if defaults:
        return defaults[key]
    return st.session_state.get(f"field_{key}", fallback)


# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-title">ChurnSight</div>
        <div class="brand-sub">Telco Customer Intelligence · ML-Powered</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card" style="margin-bottom:1.5rem">
    Enter customer attributes below and hit <b>Predict Churn</b> to get an instant inference 
    from the XGBoost model served via FastAPI. Results are returned in milliseconds.
</div>
""", unsafe_allow_html=True)

# ─── Form ─────────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # ── Demographics ──
    with col1:
        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"],
                              index=["Male","Female"].index(dv("gender","Male")))
        partner = st.selectbox("Partner", ["Yes", "No"],
                               index=["Yes","No"].index(dv("Partner","No")))
        dependents = st.selectbox("Dependents", ["Yes", "No"],
                                  index=["Yes","No"].index(dv("Dependents","No")))

        st.markdown('<div class="section-label">Phone Services</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"],
                                     index=["Yes","No"].index(dv("PhoneService","Yes")))
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"],
            index=["Yes","No","No phone service"].index(dv("MultipleLines","No"))
        )

    # ── Internet Services ──
    with col2:
        st.markdown('<div class="section-label">Internet Services</div>', unsafe_allow_html=True)
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"],
            index=["DSL","Fiber optic","No"].index(dv("InternetService","Fiber optic"))
        )
        online_sec = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"],
            index=["Yes","No","No internet service"].index(dv("OnlineSecurity","No"))
        )
        online_bk = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"],
            index=["Yes","No","No internet service"].index(dv("OnlineBackup","No"))
        )
        device_prot = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"],
            index=["Yes","No","No internet service"].index(dv("DeviceProtection","No"))
        )
        tech_sup = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"],
            index=["Yes","No","No internet service"].index(dv("TechSupport","No"))
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"],
            index=["Yes","No","No internet service"].index(dv("StreamingTV","Yes"))
        )
        streaming_mv = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"],
            index=["Yes","No","No internet service"].index(dv("StreamingMovies","Yes"))
        )

    # ── Account & Billing ──
    with col3:
        st.markdown('<div class="section-label">Account & Billing</div>', unsafe_allow_html=True)
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month","One year","Two year"].index(dv("Contract","Month-to-month"))
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"],
            index=["Yes","No"].index(dv("PaperlessBilling","Yes"))
        )
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check","Mailed check",
                   "Bank transfer (automatic)","Credit card (automatic)"].index(
                       dv("PaymentMethod","Electronic check"))
        )

        st.markdown('<div class="section-label">Numeric Features</div>', unsafe_allow_html=True)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120,
                                  value=int(dv("tenure", 1)), step=1)
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0,
                                   value=float(dv("MonthlyCharges", 85.0)), step=0.5,
                                   format="%.2f")
        total = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0,
                                 value=float(dv("TotalCharges", 85.0)), step=10.0,
                                 format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Predict Churn", use_container_width=True)


# ─── On Submit ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "gender": gender, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": online_sec,
        "OnlineBackup": online_bk, "DeviceProtection": device_prot,
        "TechSupport": tech_sup, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mv, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "tenure": int(tenure), "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    with st.spinner("Calling inference endpoint…"):
        t0 = time.time()
        result, error = call_predict(payload)
        latency_ms = int((time.time() - t0) * 1000)

    if error:
        st.error(f"**API Error:** {error}")
    else:
        prediction = result.get("prediction", "Unknown")
        is_churn = "churn" in prediction.lower() and "not" not in prediction.lower()

        res_col, meta_col = st.columns([2, 1])

        with res_col:
            if is_churn:
                st.markdown(f"""
                <div class="pred-churn">
                    <div class="pred-icon">⚠️</div>
                    <div class="pred-label" style="color:#f87171">{prediction}</div>
                    <div class="pred-desc">This customer shows high risk signals. Consider proactive retention outreach.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-safe">
                    <div class="pred-icon">✅</div>
                    <div class="pred-label" style="color:#4ade80">{prediction}</div>
                    <div class="pred-desc">Customer appears stable. Standard engagement recommended.</div>
                </div>
                """, unsafe_allow_html=True)

        with meta_col:
            st.markdown('<div class="section-label">Request Telemetry</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{latency_ms}ms</div>
                    <div class="metric-lbl">Latency</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">18</div>
                    <div class="metric-lbl">Features</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val" style="color:#818cf8">{tenure}mo</div>
                    <div class="metric-lbl">Tenure</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color:#f472b6">${monthly:.0f}</div>
                    <div class="metric-lbl">Monthly</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📦 Request Payload"):
                st.markdown(f"""
                <div class="json-block">{json.dumps(payload, indent=2)}</div>
                """, unsafe_allow_html=True)

        # ── Risk Factors Summary ──────────────────────────────────────────────
        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Risk Factor Analysis</div>', unsafe_allow_html=True)

        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append(("📋 Month-to-month contract", "High", "#ef4444"))
        if internet == "Fiber optic":
            risk_factors.append(("🌐 Fiber optic subscriber", "Elevated", "#f97316"))
        if payment == "Electronic check":
            risk_factors.append(("💳 Electronic check payment", "Elevated", "#f97316"))
        if online_sec == "No":
            risk_factors.append(("🔒 No online security", "Moderate", "#eab308"))
        if tech_sup == "No":
            risk_factors.append(("🛠️ No tech support", "Moderate", "#eab308"))
        if tenure < 12:
            risk_factors.append(("📅 New customer (< 12mo)", "Elevated", "#f97316"))
        if partner == "No" and dependents == "No":
            risk_factors.append(("👤 Single, no dependents", "Low-Moderate", "#84cc16"))

        if risk_factors:
            rf_cols = st.columns(len(risk_factors))
            for i, (label, level, color) in enumerate(risk_factors):
                rf_cols[i].markdown(f"""
                <div style="background:#0f1a2e;border:1px solid #1e2d45;border-radius:10px;
                            padding:0.8rem;text-align:center">
                    <div style="font-size:0.65rem;color:{color};text-transform:uppercase;
                                letter-spacing:2px;font-weight:600">{level}</div>
                    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tip-box">✅ No significant risk factors identified.</div>
            """, unsafe_allow_html=True)