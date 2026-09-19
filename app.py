import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


def render_html(html_string, **kwargs):
    """Render HTML robustly, working around Streamlit/Markdown's
    behavior of treating indented lines after a blank line as a
    literal code block instead of live HTML."""
    lines = [line for line in html_string.split("\n") if line.strip() != ""]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — FLAT / OPEN LAYOUT WITH REACTIVE BACKLIGHT
# ============================================================

render_html(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(70,110,190,0.06), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(255,107,107,0.035), transparent 45%),
            linear-gradient(135deg, #10131E 0%, #16233A 48%, #10131E 100%);
        color: #EAEAF2;
    }

    .main .block-container {
        max-width: 980px;
        padding-top: 2.6rem;
        padding-bottom: 5rem;
        position: relative;
        z-index: 1;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stToolbar"] { display: none; }

    /* Sidebar stays fixed open — no collapse, no animation, always visible */
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        position: relative !important;
        margin-left: 0 !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        margin-left: 0 !important;
        width: 300px !important;
        min-width: 300px !important;
    }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    section[data-testid="stSidebar"] button[kind="header"] { display: none !important; }

    /* ------------------------------------------------------
       REACTIVE BACKLIGHT — fixed overlay behind all content
       ------------------------------------------------------ */

    .glow-neutral, .glow-safe, .glow-danger {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        transition: background 1.1s ease;
    }

    .glow-neutral {
        background:
            radial-gradient(circle at 50% 0%, rgba(255,255,255,0.05), transparent 55%),
            radial-gradient(circle at 50% 100%, rgba(255,255,255,0.03), transparent 60%);
    }

    .glow-safe {
        background:
            radial-gradient(circle at 50% -5%, rgba(90,235,150,0.16), transparent 55%),
            radial-gradient(circle at 50% 105%, rgba(90,235,150,0.10), transparent 60%);
    }

    .glow-danger {
        background:
            radial-gradient(circle at 50% -5%, rgba(255,80,80,0.17), transparent 55%),
            radial-gradient(circle at 50% 105%, rgba(255,80,80,0.11), transparent 60%);
    }

    /* ------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        background: #17182A;
        border-right: 1px solid rgba(138,124,255,0.10);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .sidebar-brand {
        padding: 0 0.55rem 1.6rem 0.55rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(138,124,255,0.12);
    }

    .sidebar-brand-title {
        font-family: 'Fraunces', serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #F2F2F2;
    }

    .sidebar-brand-subtitle {
        font-size: 0.65rem;
        color: #79789C;
        margin-top: 0.35rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    div[data-testid="stSidebar"] .stRadio > label { display: none; }
    div[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }

    div[data-testid="stSidebar"] .stRadio label {
        background: transparent;
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
        transition: background 0.2s ease;
    }

    div[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(138,124,255,0.07);
    }

    div[data-testid="stSidebar"] .stRadio label p {
        color: #8887A6 !important;
        font-size: 0.82rem !important;
    }

    div[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(138,124,255,0.10);
    }

    div[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
        color: #F2F2F2 !important;
        font-weight: 600;
    }

    .sidebar-status {
        margin-top: 1.6rem;
        padding: 0 0.7rem;
    }

    .status-row { display: flex; align-items: center; gap: 0.5rem; }

    .status-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--status-color, #CFCFCF);
        box-shadow: 0 0 8px var(--status-color, rgba(255,255,255,0.5));
    }

    .status-text { color: #8E8DAE; font-size: 0.7rem; }

    /* ------------------------------------------------------
       TYPOGRAPHY / HERO — flat, centered, no boxes
       ------------------------------------------------------ */

    .kicker {
        text-align: center;
        color: #82819F;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        text-align: center;
        font-family: 'Fraunces', serif;
        font-size: 2.9rem;
        font-weight: 600;
        line-height: 1.18;
        letter-spacing: -0.01em;
        background: linear-gradient(180deg, #F2F5FA 0%, #C7D6EE 60%, #6E93D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.9rem;
    }

    .hero-title em { font-style: italic; font-weight: 500; color: #5C82C7; -webkit-text-fill-color: #5C82C7; }

    .hero-subtitle {
        text-align: center;
        color: #9493AF;
        font-size: 0.94rem;
        line-height: 1.75;
        max-width: 560px;
        margin: 0 auto 2.4rem auto;
    }

    .hero-icon { display: flex; justify-content: center; margin-bottom: 1.4rem; }

    .section-title {
        font-family: 'Fraunces', serif;
        font-style: italic;
        color: #C7C3EC;
        font-size: 1.25rem;
        font-weight: 500;
        text-align: center;
        margin: 2.6rem 0 1.5rem 0;
    }

    .divider {
        height: 1px;
        width: 100%;
        max-width: 460px;
        margin: 2.2rem auto;
        background: linear-gradient(90deg, transparent, rgba(138,124,255,0.28), transparent);
    }

    /* ------------------------------------------------------
       FEATURE ROW — icon + text, no cards
       ------------------------------------------------------ */

    .feature {
        text-align: center;
        padding: 0.5rem 1rem;
    }

    .feature-icon { display: flex; justify-content: center; margin-bottom: 0.8rem; }

    .feature-title {
        color: #D6D3EE;
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .feature-text {
        color: #83829F;
        font-size: 0.76rem;
        line-height: 1.65;
    }

    /* ------------------------------------------------------
       STAT ROW — flat figures with thin dividers, not boxes
       ------------------------------------------------------ */

    .stat-row {
        display: flex;
        justify-content: center;
        align-items: stretch;
        gap: 0;
        margin: 1.4rem 0 2.2rem 0;
        flex-wrap: wrap;
    }

    .stat-item {
        text-align: center;
        padding: 0 2.2rem;
        border-left: 1px solid rgba(138,124,255,0.14);
    }

    .stat-item:first-child { border-left: none; }

    .stat-value {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 600;
        color: #EDEBFA;
    }

    .stat-label {
        color: #7B7A98;
        font-size: 0.66rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.35rem;
    }

    /* ------------------------------------------------------
       RESULT BLOCK — flat, centered, color-coded by outcome
       ------------------------------------------------------ */

    .result-label {
        text-align: center;
        color: #82819F;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.6rem;
    }

    .result-value {
        text-align: center;
        font-family: 'Fraunces', serif;
        font-size: 3.4rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .result-verdict {
        text-align: center;
        font-family: 'Fraunces', serif;
        font-size: 1.7rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }

    .result-safe { color: #7BEBA6; text-shadow: 0 0 22px rgba(123,235,166,0.25); }
    .result-danger { color: #FF7A7A; text-shadow: 0 0 22px rgba(255,122,122,0.25); }

    .recommendation {
        text-align: center;
        color: #9997B4;
        font-size: 0.82rem;
        line-height: 1.7;
        max-width: 480px;
        margin: 0 auto;
    }

    /* ------------------------------------------------------
       INPUTS
       ------------------------------------------------------ */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: #1F2036 !important;
        color: #E7E6F2 !important;
        border: 1px solid #35365A !important;
        border-radius: 9px !important;
    }

    label { color: #A2A0BE !important; font-size: 0.76rem !important; font-weight: 500 !important; }

    div[data-testid="stSlider"] div[data-baseweb="slider"] {
        filter: hue-rotate(240deg) saturate(1.5) brightness(1.25);
    }

    /* ------------------------------------------------------
       BUTTONS — soft pill, centered feel
       ------------------------------------------------------ */

    .stButton, .stDownloadButton { display: flex; justify-content: center; }

    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(180deg, #F0F0F0 0%, #B9B9B9 100%);
        color: #14151F;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 2.1rem;
        font-weight: 700;
        font-size: 0.78rem;
        box-shadow: 0 6px 22px rgba(0,0,0,0.35), 0 0 18px rgba(138,124,255,0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.4), 0 0 24px rgba(138,124,255,0.20);
        background: linear-gradient(180deg, #FFFFFF 0%, #CFCFCF 100%);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(138,124,255,0.25);
        border-radius: 12px;
        background: transparent;
        padding: 0.6rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(138,124,255,0.14);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        background: rgba(138,124,255,0.05);
        border: 1px solid rgba(138,124,255,0.14);
        border-radius: 10px;
        color: #C6C4DE;
    }

    /* Soft ambient glow around the batch-analysis charts */
    div[data-testid="stPlotlyChart"] {
        filter: drop-shadow(0 0 16px rgba(138,124,255,0.10));
    }

    .footer {
        text-align: center;
        color: #55546E;
        font-size: 0.66rem;
        padding: 3.5rem 0 0.5rem 0;
        letter-spacing: 0.03em;
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    model = joblib.load("predictive_maintenance_model.pkl")
    threshold = joblib.load("failure_threshold.pkl")
    return model, threshold


model, threshold = load_model()


# ============================================================
# REACTIVE BACKLIGHT STATE
# ============================================================

if "risk_status" not in st.session_state:
    st.session_state.risk_status = "neutral"

glow_placeholder = st.empty()


def render_glow(status):
    class_map = {"neutral": "glow-neutral", "safe": "glow-safe", "danger": "glow-danger"}
    glow_placeholder.markdown(
        f'<div class="{class_map.get(status, "glow-neutral")}"></div>',
        unsafe_allow_html=True
    )


render_glow(st.session_state.risk_status)

_status_color = {
    "neutral": "#CFCFCF",
    "safe": "#7BEBA6",
    "danger": "#FF7A7A"
}[st.session_state.risk_status]

_status_text = {
    "neutral": "Model ready · awaiting input",
    "safe": "Last check: conditions nominal",
    "danger": "Last check: elevated risk"
}[st.session_state.risk_status]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Predictive Maintenance</div>
            <div class="sidebar-brand-subtitle">Machine Risk Monitoring</div>
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        ["Predict", "Batch Predict"],
        label_visibility="collapsed"
    )

    render_html(
        f"""
        <div class="sidebar-status" style="--status-color: {_status_color};">
            <div class="status-row">
                <div class="status-dot"></div>
                <div class="status-text">{_status_text}</div>
            </div>
        </div>
        """
    )


# ============================================================
# SHARED ICON (decorative gauge)
# ============================================================

GAUGE_ICON = """
<div class="hero-icon">
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 44a22 22 0 1 1 44 0" stroke="#5C5A80" stroke-width="1.6" stroke-linecap="round"/>
        <path d="M32 44 L42 30" stroke="#9C93E0" stroke-width="1.8" stroke-linecap="round"/>
        <circle cx="32" cy="44" r="3" fill="#E6E4F5"/>
        <circle cx="10" cy="44" r="1.6" fill="#FDBA43"/>
        <circle cx="54" cy="44" r="1.6" fill="#FF6B6B"/>
        <circle cx="32" cy="22" r="1.6" fill="#8A7CFF"/>
    </svg>
</div>
"""


# ============================================================
# PAGE: PREDICT
# ============================================================

if page == "Predict":

    render_html('<div class="kicker">Operational Assessment</div>')
    render_html(GAUGE_ICON)
    render_html(
        '<div class="hero-title">Assess machine conditions,<br><em>catch failure early.</em></div>'
    )
    render_html(
        '<div class="hero-subtitle">Enter the current operating conditions below and the '
        'model will estimate the probability of failure using a trained Random Forest '
        'classifier.</div>'
    )

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temperature = st.slider(
            "Air Temperature [K]", min_value=250.0, max_value=350.0,
            value=298.1, step=0.1, format="%.1f"
        )
        process_temperature = st.slider(
            "Process Temperature [K]", min_value=250.0, max_value=400.0,
            value=308.6, step=0.1, format="%.1f"
        )

    with col2:
        rotational_speed = st.slider(
            "Rotational Speed [rpm]", min_value=1000, max_value=3000,
            value=1500, step=10
        )
        torque = st.slider(
            "Torque [Nm]", min_value=0.0, max_value=100.0,
            value=40.0, step=0.1, format="%.1f"
        )
        tool_wear = st.slider(
            "Tool Wear [min]", min_value=0, max_value=300,
            value=100, step=1
        )

    render_html('<div class="divider"></div>')

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        run_clicked = st.button("Assess Failure Risk")

    if run_clicked:

        input_data = pd.DataFrame({
            "Type": [machine_type],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear]
        })

        probability = model.predict_proba(input_data)[0, 1]
        prediction = int(probability >= threshold)

        st.session_state.risk_status = "danger" if prediction == 1 else "safe"
        render_glow(st.session_state.risk_status)

        verdict_class = "result-danger" if prediction == 1 else "result-safe"
        verdict_text = "Elevated Failure Risk" if prediction == 1 else "No Elevated Risk"
        recommendation_text = (
            "The model identifies elevated failure risk for these operating "
            "conditions. Recommend investigation by maintenance personnel."
            if prediction == 1 else
            "These operating conditions are not classified as elevated risk "
            "at the current decision threshold."
        )

        render_html('<div class="divider"></div>')
        render_html('<div class="result-label">Failure Probability</div>')
        render_html(f'<div class="result-value {verdict_class}">{probability * 100:.1f}%</div>')
        render_html(f'<div class="result-verdict {verdict_class}">{verdict_text}</div>')
        render_html(f'<div class="recommendation">{recommendation_text}</div>')


# ============================================================
# PAGE: BATCH PREDICT
# ============================================================

elif page == "Batch Predict":

    render_html('<div class="kicker">Bulk Analysis</div>')
    render_html(GAUGE_ICON)
    render_html(
        '<div class="hero-title">Monitor a fleet,<br><em>not just one machine.</em></div>'
    )
    render_html(
        '<div class="hero-subtitle">Upload a CSV of machine observations to generate '
        'failure-risk predictions in bulk and export an enriched dataset.</div>'
    )

    required_columns = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]

    uploaded_file = st.file_uploader("Upload machine data", type=["csv"])

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_columns = [c for c in required_columns if c not in batch_df.columns]

            if missing_columns:
                st.error("The uploaded CSV is missing required columns: " + ", ".join(missing_columns))

            else:
                st.success(f"File loaded successfully: {len(batch_df):,} observations.")

                prediction_input = batch_df[required_columns].copy()
                probabilities = model.predict_proba(prediction_input)[:, 1]
                predictions = (probabilities >= threshold).astype(int)

                results_df = batch_df.copy()
                results_df["Failure Probability"] = probabilities
                results_df["Predicted Failure"] = predictions

                total_records = len(results_df)
                predicted_failures = int(predictions.sum())
                average_probability = probabilities.mean() * 100

                st.session_state.risk_status = "danger" if predicted_failures > 0 else "safe"
                render_glow(st.session_state.risk_status)

                render_html('<div class="divider"></div>')

                render_html(
                    f"""
                    <div class="stat-row">
                        <div class="stat-item">
                            <div class="stat-value">{total_records:,}</div>
                            <div class="stat-label">Observations</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{predicted_failures:,}</div>
                            <div class="stat-label">Predicted Failures</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{average_probability:.1f}%</div>
                            <div class="stat-label">Avg. Probability</div>
                        </div>
                    </div>
                    """
                )

                render_html('<div class="section-title">Visual Breakdown</div>')

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:

                    risk_pct = predicted_failures / total_records * 100
                    gauge_color = "#FF6B6B" if predicted_failures > 0 else "#5AEB96"
                    gauge_track = "rgba(255,107,107,0.12)" if predicted_failures > 0 else "rgba(90,235,150,0.12)"

                    donut_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_pct,
                        number=dict(
                            suffix="%",
                            font=dict(family="Fraunces", size=34, color="#EDEBFA")
                        ),
                        gauge=dict(
                            shape="angular",
                            axis=dict(range=[0, 100], visible=False),
                            bar=dict(color=gauge_color, thickness=0.32),
                            bgcolor=gauge_track,
                            borderwidth=0,
                        )
                    ))

                    donut_fig.update_layout(
                        title=dict(text="Observations at Risk", font=dict(color="#C9C7DE", size=13, family="Inter")),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#C9C7DE", family="Inter"),
                        margin=dict(t=60, b=20, l=30, r=30),
                        height=340
                    )

                    st.plotly_chart(donut_fig, use_container_width=True)

                with chart_col2:

                    type_summary = (
                        results_df.groupby("Type")["Failure Probability"]
                        .mean()
                        .reindex(["L", "M", "H"])
                        .dropna() * 100
                    )

                    palette = {"L": "#8A7CFF", "M": "#FDBA43", "H": "#FF6B6B"}
                    glow_line = {"L": "#B7ACFF", "M": "#FFD48A", "H": "#FF9E9E"}
                    bar_colors = [palette.get(t, "#8887A6") for t in type_summary.index]
                    line_colors = [glow_line.get(t, "#B0AFCB") for t in type_summary.index]

                    bar_fig = go.Figure()

                    bar_fig.add_trace(go.Bar(
                        x=type_summary.index.tolist(),
                        y=type_summary.values.tolist(),
                        width=0.42,
                        marker=dict(
                            color=bar_colors,
                            opacity=0.9,
                            line=dict(color=line_colors, width=1.5)
                        ),
                        text=[f"{v:.1f}%" for v in type_summary.values],
                        textposition="outside",
                        textfont=dict(color="#C9C7DE"),
                        hovertemplate="Type %{x}: %{y:.1f}%<extra></extra>",
                        showlegend=False
                    ))

                    bar_fig.add_trace(go.Scatter(
                        x=type_summary.index.tolist(),
                        y=type_summary.values.tolist(),
                        mode="markers",
                        marker=dict(color=line_colors, size=15, line=dict(width=0)),
                        hoverinfo="skip",
                        showlegend=False
                    ))

                    bar_fig.update_layout(
                        title=dict(text="Avg. Failure Probability by Machine Type",
                                   font=dict(color="#C9C7DE", size=13, family="Inter")),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#C9C7DE", family="Inter"),
                        xaxis=dict(showgrid=False, title="Machine Type"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(138,124,255,0.08)",
                                   title="Failure Probability (%)"),
                        margin=dict(t=50, b=20, l=10, r=10),
                        height=340
                    )

                    st.plotly_chart(bar_fig, use_container_width=True)

                render_html('<div class="section-title">Prediction Results</div>')

                display_df = results_df.copy()
                display_df["Failure Probability"] = (display_df["Failure Probability"] * 100).round(2)
                display_df["Predicted Failure"] = display_df["Predicted Failure"].map({
                    0: "No Failure", 1: "Potential Failure"
                })

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                render_html('<div class="divider"></div>')

                csv_data = results_df.to_csv(index=False).encode("utf-8")

                _, dl_col, _ = st.columns([1, 1, 1])
                with dl_col:
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_data,
                        file_name="predictive_maintenance_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Unable to process the uploaded file: {str(e)}")

    else:
        render_html('<div class="divider"></div>')
        render_html(
            """
            <div class="feature">
                <div class="feature-title">Required CSV columns</div>
                <div class="feature-text">
                    Type, Air temperature [K], Process temperature [K],
                    Rotational speed [rpm], Torque [Nm], Tool wear [min]
                </div>
            </div>
            """
        )


render_html(
    """
    <div class="footer">
        Predictive Maintenance &nbsp;·&nbsp; Machine Failure Risk Monitoring
    </div>
    """
)