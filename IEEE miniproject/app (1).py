import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(
    page_title="NeuroScan | Alzheimer's Risk Prediction System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

STEP_NAMES = ["Welcome", "Patient Details", "Health Assessment", "Memory Test", "Speech Analysis", "Processing", "Results"]


def init_state():
    defaults = {
        "page": "welcome",
        "step": 0,
        "theme_mode": "dark",
        "patient_name": "",
        "patient_age": 60,
        "patient_gender": "Male",
        "patient_concern": "",
        "patient_id": "NS-" + time.strftime("%H%M%S"),
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def inject_theme():
    dark = st.session_state.theme_mode == "dark"

    if dark:
        bg_base = "#0f172a"
        bg_card = "rgba(255,255,255,0.035)"
        card_border = "rgba(37,99,235,0.22)"
        text_primary = "#e8eefc"
        text_secondary = "rgba(200,215,245,0.62)"
        text_faint = "rgba(200,215,245,0.4)"
        accent = "#2563eb"
        accent_2 = "#06b6d4"
        accent_3 = "#7c8cf8"
        input_bg = "rgba(37,99,235,0.07)"
        input_border = "rgba(37,99,235,0.28)"
        divider = "rgba(37,99,235,0.22)"
        shadow = "rgba(0,0,0,0.35)"
    else:
        bg_base = "#eef3fb"
        bg_card = "rgba(255,255,255,0.78)"
        card_border = "rgba(37,99,235,0.18)"
        text_primary = "#0f172a"
        text_secondary = "rgba(30,41,59,0.68)"
        text_faint = "rgba(30,41,59,0.45)"
        accent = "#2563eb"
        accent_2 = "#0891b2"
        accent_3 = "#4f5fce"
        input_bg = "rgba(37,99,235,0.05)"
        input_border = "rgba(37,99,235,0.25)"
        divider = "rgba(37,99,235,0.18)"
        shadow = "rgba(30,41,59,0.12)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{ font-family: 'Inter', sans-serif; }}

    [data-testid="stApp"] {{
        background: {bg_base} !important;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }}

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main, section.main {{
        background: transparent !important;
    }}

    .block-container {{
        background: transparent !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 880px !important;
        position: relative;
        z-index: 2;
    }}

    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

    .ns-bg {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
    }}

    .ns-topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
    }}

    .ns-logo {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: {text_primary};
    }}

    .ns-logo .dot {{
        width: 9px; height: 9px;
        border-radius: 50%;
        background: linear-gradient(135deg, {accent}, {accent_2});
        box-shadow: 0 0 10px {accent};
    }}

    .sub-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: {accent_2};
        margin-bottom: 0.3rem;
        display: block;
    }}

    .section-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        color: {text_primary};
        margin: 0 0 0.7rem 0;
        font-weight: 600;
    }}

    .sub-desc {{
        font-size: 0.85rem;
        color: {text_secondary};
        margin: 0.1rem 0 0.9rem;
        line-height: 1.55;
    }}

    .hero-block {{
        text-align: center;
        padding: 1.6rem 1rem 1.2rem;
    }}

    .brain-icon {{
        font-size: 4rem;
        display: block;
        margin-bottom: 0.6rem;
        animation: brainGlow 3.5s ease-in-out infinite;
    }}

    @keyframes brainGlow {{
        0%, 100% {{ filter: drop-shadow(0 0 14px rgba(37,99,235,0.45)); transform: scale(1); }}
        50%       {{ filter: drop-shadow(0 0 30px rgba(6,182,212,0.75)); transform: scale(1.05); }}
    }}

    .hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: {text_primary};
        letter-spacing: -0.01em;
        line-height: 1.18;
        margin-bottom: 0.55rem;
    }}

    .hero-title span {{
        background: linear-gradient(135deg, {accent} 0%, {accent_2} 60%, {accent_3} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .hero-subtitle {{
        font-size: 0.95rem;
        color: {text_secondary};
        font-weight: 400;
        line-height: 1.65;
        max-width: 520px;
        margin: 0 auto;
    }}

    .ns-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {divider}, transparent);
        margin: 1.1rem 0;
    }}

    .feature-card {{
        background: {bg_card};
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 1.2rem 1.1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 6px 20px {shadow};
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }}

    .feature-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 28px {shadow};
    }}

    .feature-card .fi {{
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        display: block;
    }}

    .feature-card .ft {{
        font-weight: 600;
        font-size: 0.92rem;
        color: {text_primary};
        margin-bottom: 0.25rem;
        font-family: 'Outfit', sans-serif;
    }}

    .feature-card .fd {{
        font-size: 0.76rem;
        color: {text_secondary};
        line-height: 1.4;
    }}

    .progress-steps {{
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }}

    .step-pill {{
        flex: 1;
        min-width: 36px;
        height: 5px;
        border-radius: 4px;
        background: {input_bg};
        position: relative;
        overflow: hidden;
    }}

    .step-pill.done {{
        background: linear-gradient(90deg, {accent}, {accent_2});
    }}

    .step-pill.active {{
        background: linear-gradient(90deg, {accent}, {accent_2});
        box-shadow: 0 0 10px {accent};
    }}

    .step-label-row {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }}

    .step-counter {{
        font-size: 0.74rem;
        font-weight: 700;
        color: {accent_2};
        letter-spacing: 0.05em;
    }}

    .step-name {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: {text_primary};
    }}

    label, .stSelectbox label, .stNumberInput label, .stRadio label,
    .stCheckbox label, .stTextInput label, .stTextArea label {{
        color: {text_primary} !important;
        opacity: 0.88;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }}

    .stTextInput input, .stNumberInput input, .stTextArea textarea {{
        background: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        caret-color: {accent} !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
        outline: none !important;
    }}

    .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: {text_faint} !important;
    }}

    [data-baseweb="select"] > div, .stSelectbox > div > div {{
        background: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
    }}

    [data-baseweb="select"] svg {{ color: {accent_2} !important; }}

    [data-baseweb="popover"] div, [data-baseweb="menu"] {{
        background: {bg_card} !important;
        border: 1px solid {input_border} !important;
        border-radius: 12px !important;
    }}

    [data-baseweb="option"]:hover {{ background: rgba(37,99,235,0.14) !important; }}
    [data-baseweb="option"] {{ color: {text_primary} !important; }}

    .stRadio > div {{ gap: 0.45rem; flex-wrap: wrap; }}

    div[role="radiogroup"] label {{
        background: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 9px !important;
        padding: 0.42rem 0.9rem !important;
        color: {text_primary} !important;
        opacity: 0.9;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 0.86rem !important;
    }}

    div[role="radiogroup"] label:hover {{
        background: rgba(37,99,235,0.16) !important;
        border-color: {accent} !important;
    }}

    .stNumberInput button {{
        background: {input_bg} !important;
        border: 1px solid {input_border} !important;
        color: {accent_2} !important;
        border-radius: 8px !important;
    }}

    .stCheckbox label {{ color: {text_primary} !important; opacity: 0.9; }}

    .stButton > button {{
        background: linear-gradient(135deg, {accent} 0%, {accent_2} 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.93rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.65rem 1.8rem !important;
        width: 100% !important;
        cursor: pointer !important;
        box-shadow: 0 4px 18px rgba(37,99,235,0.35) !important;
        transition: transform 0.18s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.18s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 26px rgba(6,182,212,0.45) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0px) scale(0.98) !important;
        box-shadow: 0 2px 10px rgba(37,99,235,0.4) !important;
        transition: transform 0.08s ease, box-shadow 0.08s ease !important;
    }}

    .stButton > button::after {{
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(105deg, transparent 35%, rgba(255,255,255,0.22) 50%, transparent 65%);
        transform: translateX(-150%);
        transition: transform 0.55s ease;
    }}

    .stButton > button:hover::after {{ transform: translateX(150%); }}

    .stDownloadButton > button {{
        background: {bg_card} !important;
        color: {accent_2} !important;
        border: 1.5px solid {accent_2} !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: none !important;
    }}

    .stDownloadButton > button:hover {{
        background: rgba(6,182,212,0.1) !important;
        transform: translateY(-1px) !important;
    }}

    div[data-testid="stAlert"] {{
        background: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 12px !important;
    }}

    div[data-testid="stAlert"] p {{ color: {text_primary} !important; }}

    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {accent}, {accent_2}) !important;
        border-radius: 10px !important;
    }}

    .score-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(37,99,235,0.12);
        border: 1px solid rgba(37,99,235,0.3);
        border-radius: 50px;
        padding: 0.28rem 0.95rem;
        font-size: 0.98rem;
        font-weight: 700;
        color: {accent_2};
        letter-spacing: 0.02em;
    }}

    .transcript-bubble {{
        background: {input_bg};
        border: 1px solid {input_border};
        border-radius: 14px 14px 14px 4px;
        padding: 0.8rem 1.1rem;
        margin: 0.6rem 0 0.7rem;
        font-size: 0.87rem;
        color: {text_primary};
        opacity: 0.92;
        font-style: italic;
        line-height: 1.55;
        max-width: 90%;
    }}

    .word-badges {{
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.6rem 0 0.8rem;
    }}

    .word-badge {{
        background: rgba(37,99,235,0.1);
        border: 1px solid rgba(37,99,235,0.3);
        border-radius: 8px;
        padding: 0.32rem 0.85rem;
        color: {accent_2};
        font-weight: 600;
        font-size: 0.93rem;
        letter-spacing: 0.07em;
    }}

    .metric-row {{
        display: flex;
        gap: 0.8rem;
        margin: 0.8rem 0;
        flex-wrap: wrap;
    }}

    .metric-chip {{
        flex: 1;
        min-width: 100px;
        background: {bg_card};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 0.85rem;
        text-align: center;
    }}

    .metric-chip .val {{
        font-size: 1.45rem;
        font-weight: 700;
        color: {accent_2};
        font-family: 'Outfit', sans-serif;
    }}

    .metric-chip .lbl {{
        font-size: 0.68rem;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.18rem;
    }}

    .patient-card {{
        background: {bg_card};
        border: 1px solid {card_border};
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 6px 24px {shadow};
    }}

    .patient-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.7rem;
        margin-top: 0.8rem;
    }}

    .patient-field .pl {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {text_secondary};
        margin-bottom: 0.15rem;
    }}

    .patient-field .pv {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {text_primary};
        font-family: 'Outfit', sans-serif;
    }}

    .risk-card {{
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-top: 0.8rem;
        border: 1.5px solid;
        backdrop-filter: blur(10px);
    }}

    .risk-card.high   {{ border-color: rgba(239,68,68,0.5); background: rgba(239,68,68,0.07); }}
    .risk-card.medium {{ border-color: rgba(245,158,11,0.5); background: rgba(245,158,11,0.07); }}
    .risk-card.low    {{ border-color: rgba(16,185,129,0.5); background: rgba(16,185,129,0.07); }}

    .risk-title {{ font-family:'Outfit',sans-serif; font-size:1.6rem; margin-bottom:0.25rem; font-weight:700; }}
    .risk-card.high .risk-title   {{ color: #ef4444; }}
    .risk-card.medium .risk-title {{ color: #f59e0b; }}
    .risk-card.low .risk-title    {{ color: #10b981; }}

    .risk-rec {{
        font-size: 0.87rem;
        color: {text_primary};
        opacity: 0.85;
        line-height: 1.65;
        margin-top: 0.55rem;
    }}

    .emergency-banner {{
        background: rgba(239,68,68,0.12);
        border: 1.5px solid rgba(239,68,68,0.5);
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        animation: emergencyPulse 1.8s ease-in-out infinite;
    }}

    @keyframes emergencyPulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }}
        50%       {{ box-shadow: 0 0 0 8px rgba(239,68,68,0); }}
    }}

    .emergency-banner .et {{
        font-weight: 700;
        color: #ef4444;
        font-size: 0.95rem;
    }}

    .emergency-banner .ed {{
        font-size: 0.82rem;
        color: {text_primary};
        opacity: 0.8;
    }}

    .processing-card {{
        background: {bg_card};
        border: 1px solid {card_border};
        border-radius: 20px;
        padding: 2.4rem 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px {shadow};
    }}

    .processing-brain {{
        font-size: 4.5rem;
        display: inline-block;
        animation: procSpin 2.4s linear infinite, brainGlow 3.5s ease-in-out infinite;
        margin-bottom: 1rem;
    }}

    @keyframes procSpin {{
        0%   {{ transform: rotate(0deg) scale(1); }}
        50%  {{ transform: rotate(180deg) scale(1.08); }}
        100% {{ transform: rotate(360deg) scale(1); }}
    }}

    .proc-step {{
        font-size: 0.88rem;
        color: {text_secondary};
        margin: 0.3rem 0;
        opacity: 0.5;
        transition: opacity 0.3s ease;
    }}

    .proc-step.active {{
        opacity: 1;
        color: {accent_2};
        font-weight: 600;
    }}

    .proc-step.complete {{
        opacity: 0.85;
        color: #10b981;
    }}

    .mic-wrapper {{
        background: {input_bg};
        border: 1px solid {input_border};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin: 0.6rem 0;
        text-align: center;
    }}

    .mic-pulse-ring {{
        position: relative;
        width: 64px;
        height: 64px;
        margin: 0 auto 0.7rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .mic-pulse-ring .ring {{
        position: absolute;
        width: 100%; height: 100%;
        border-radius: 50%;
        border: 2px solid {accent_2};
        opacity: 0;
        animation: micPulse 1.8s ease-out infinite;
    }}

    .mic-pulse-ring .ring:nth-child(2) {{ animation-delay: 0.6s; }}
    .mic-pulse-ring .ring:nth-child(3) {{ animation-delay: 1.2s; }}

    @keyframes micPulse {{
        0%   {{ transform: scale(0.5); opacity: 0.8; }}
        100% {{ transform: scale(1.6); opacity: 0; }}
    }}

    .mic-icon-static {{
        font-size: 1.8rem;
        position: relative;
        z-index: 2;
    }}

    .waveform {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        height: 36px;
        margin: 0.6rem 0;
    }}

    .waveform span {{
        width: 3px;
        background: linear-gradient(180deg, {accent_2}, {accent});
        border-radius: 2px;
        animation: waveBar 1.2s ease-in-out infinite;
    }}

    .waveform span:nth-child(1) {{ height: 30%; animation-delay: 0s; }}
    .waveform span:nth-child(2) {{ height: 60%; animation-delay: 0.1s; }}
    .waveform span:nth-child(3) {{ height: 90%; animation-delay: 0.2s; }}
    .waveform span:nth-child(4) {{ height: 50%; animation-delay: 0.3s; }}
    .waveform span:nth-child(5) {{ height: 75%; animation-delay: 0.4s; }}
    .waveform span:nth-child(6) {{ height: 40%; animation-delay: 0.5s; }}
    .waveform span:nth-child(7) {{ height: 65%; animation-delay: 0.6s; }}
    .waveform span:nth-child(8) {{ height: 35%; animation-delay: 0.7s; }}

    @keyframes waveBar {{
        0%, 100% {{ transform: scaleY(0.4); }}
        50%       {{ transform: scaleY(1); }}
    }}

    .gauge-wrap {{
        position: relative;
        width: 220px;
        height: 220px;
        margin: 0.5rem auto 0.8rem;
    }}

    .gauge-center {{
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }}

    .gauge-center .gv {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
    }}

    .gauge-center .gl {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {text_secondary};
        margin-top: 0.2rem;
    }}

    .bar-chart-row {{
        display: flex;
        align-items: flex-end;
        gap: 1.4rem;
        height: 160px;
        margin: 1rem 0 0.4rem;
        padding: 0 0.4rem;
    }}

    .bar-chart-col {{
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        height: 100%;
    }}

    .bar-chart-bar {{
        width: 38px;
        border-radius: 8px 8px 0 0;
        transition: height 0.6s ease;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.15);
    }}

    .bar-chart-val {{
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: {text_primary};
    }}

    .bar-chart-lbl {{
        font-size: 0.68rem;
        color: {text_secondary};
        margin-top: 0.4rem;
        text-align: center;
    }}

    iframe {{
        background: transparent !important;
        border-radius: 10px !important;
        border: none !important;
    }}

    [data-testid="stHorizontalBlock"] {{ gap: 0.8rem; }}

    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: rgba(120,140,180,0.08); }}
    ::-webkit-scrollbar-thumb {{ background: rgba(37,99,235,0.3); border-radius: 10px; }}

    h1, h2, h3 {{
        color: {text_primary} !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }}

    [data-testid="stCaptionContainer"], .stCaption {{
        color: {text_secondary} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_background_svg():
    if st.session_state.theme_mode == "dark":
        base = "#0a1228"
        c1, c2, c3, c4 = "#2563eb", "#06b6d4", "#7c8cf8", "#e8eefc"
        glow_op = "0.5"
        neuron_op = "0.85"
        synapse_op_lo, synapse_op_hi = "0.08", "0.3"
        wave_op = "0.35"
        helix_op = "0.18"
        ring_op = "0.14"
        particle_op = "0.8"
    else:
        base = "#eef3fb"
        c1, c2, c3, c4 = "#2563eb", "#0891b2", "#94a3f0", "#3b4a6b"
        glow_op = "0.2"
        neuron_op = "0.55"
        synapse_op_lo, synapse_op_hi = "0.05", "0.16"
        wave_op = "0.18"
        helix_op = "0.1"
        ring_op = "0.08"
        particle_op = "0.5"

    st.markdown(f"""
    <svg class="ns-bg" viewBox="0 0 1400 900" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
      <defs>
        <radialGradient id="bgGlow1" cx="20%" cy="20%" r="55%">
          <stop offset="0%" stop-color="{c1}" stop-opacity="{glow_op}"/>
          <stop offset="100%" stop-color="{base}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="bgGlow2" cx="82%" cy="78%" r="55%">
          <stop offset="0%" stop-color="{c2}" stop-opacity="{glow_op}"/>
          <stop offset="100%" stop-color="{base}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="bgGlow3" cx="55%" cy="50%" r="45%">
          <stop offset="0%" stop-color="{c3}" stop-opacity="{glow_op}"/>
          <stop offset="100%" stop-color="{base}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="neuronCore" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{c4}" stop-opacity="0.95"/>
          <stop offset="40%" stop-color="{c2}" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="{c2}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="neuronCore2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{c4}" stop-opacity="0.95"/>
          <stop offset="40%" stop-color="{c1}" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="{c1}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="neuronCore3" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{c4}" stop-opacity="0.95"/>
          <stop offset="40%" stop-color="{c3}" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="{c3}" stop-opacity="0"/>
        </radialGradient>
        <filter id="softBlur" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="2.4" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="particleBlur" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="1.1"/>
        </filter>
      </defs>

      <rect width="1400" height="900" fill="{base}"/>
      <rect width="1400" height="900" fill="url(#bgGlow1)">
        <animate attributeName="opacity" values="0.65;1;0.65" dur="14s" repeatCount="indefinite"/>
      </rect>
      <rect width="1400" height="900" fill="url(#bgGlow2)">
        <animate attributeName="opacity" values="1;0.6;1" dur="17s" repeatCount="indefinite"/>
      </rect>
      <rect width="1400" height="900" fill="url(#bgGlow3)">
        <animate attributeName="opacity" values="0.5;0.9;0.5" dur="20s" repeatCount="indefinite"/>
      </rect>

      <g opacity="{helix_op}" stroke="{c3}" stroke-width="1.4" fill="none" stroke-linecap="round">
        <path d="M 60,40 C 110,90 10,140 60,190 C 110,240 10,290 60,340 C 110,390 10,440 60,490 C 110,540 10,590 60,640">
          <animate attributeName="d"
            values="M 60,40 C 110,90 10,140 60,190 C 110,240 10,290 60,340 C 110,390 10,440 60,490 C 110,540 10,590 60,640;
                    M 60,40 C 10,90 110,140 60,190 C 10,240 110,290 60,340 C 10,390 110,440 60,490 C 10,540 110,590 60,640;
                    M 60,40 C 110,90 10,140 60,190 C 110,240 10,290 60,340 C 110,390 10,440 60,490 C 110,540 10,590 60,640"
            dur="22s" repeatCount="indefinite"/>
        </path>
        <path d="M 60,40 C 10,90 110,140 60,190 C 10,240 110,290 60,340 C 10,390 110,440 60,490 C 10,540 110,590 60,640">
          <animate attributeName="d"
            values="M 60,40 C 10,90 110,140 60,190 C 10,240 110,290 60,340 C 10,390 110,440 60,490 C 10,540 110,590 60,640;
                    M 60,40 C 110,90 10,140 60,190 C 110,240 10,290 60,340 C 110,390 10,440 60,490 C 110,540 10,590 60,640;
                    M 60,40 C 10,90 110,140 60,190 C 10,240 110,290 60,340 C 10,390 110,440 60,490 C 10,540 110,590 60,640"
            dur="22s" repeatCount="indefinite"/>
        </path>
      </g>

      <g opacity="{helix_op}" stroke="{c2}" stroke-width="1.4" fill="none" stroke-linecap="round">
        <path d="M 1340,260 C 1390,310 1290,360 1340,410 C 1390,460 1290,510 1340,560 C 1390,610 1290,660 1340,710">
          <animate attributeName="d"
            values="M 1340,260 C 1390,310 1290,360 1340,410 C 1390,460 1290,510 1340,560 C 1390,610 1290,660 1340,710;
                    M 1340,260 C 1290,310 1390,360 1340,410 C 1290,460 1390,510 1340,560 C 1290,610 1390,660 1340,710;
                    M 1340,260 C 1390,310 1290,360 1340,410 C 1390,460 1290,510 1340,560 C 1390,610 1290,660 1340,710"
            dur="26s" repeatCount="indefinite"/>
        </path>
        <path d="M 1340,260 C 1290,310 1390,360 1340,410 C 1290,460 1390,510 1340,560 C 1290,610 1390,660 1340,710">
          <animate attributeName="d"
            values="M 1340,260 C 1290,310 1390,360 1340,410 C 1290,460 1390,510 1340,560 C 1290,610 1390,660 1340,710;
                    M 1340,260 C 1390,310 1290,360 1340,410 C 1390,460 1290,510 1340,560 C 1390,610 1290,660 1340,710;
                    M 1340,260 C 1290,310 1390,360 1340,410 C 1290,460 1390,510 1340,560 C 1290,610 1390,660 1340,710"
            dur="26s" repeatCount="indefinite"/>
        </path>
      </g>

      <g fill="none" stroke="{c1}" opacity="{ring_op}">
        <circle cx="220" cy="700" r="60" stroke-width="1.2">
          <animate attributeName="r" values="50;75;50" dur="12s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.05;0.18;0.05" dur="12s" repeatCount="indefinite"/>
        </circle>
        <circle cx="220" cy="700" r="95" stroke-width="1">
          <animate attributeName="r" values="85;110;85" dur="12s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.03;0.12;0.03" dur="12s" repeatCount="indefinite"/>
        </circle>
        <circle cx="1170" cy="160" r="55" stroke-width="1.2">
          <animate attributeName="r" values="45;68;45" dur="15s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.05;0.16;0.05" dur="15s" repeatCount="indefinite"/>
        </circle>
        <circle cx="1170" cy="160" r="85" stroke-width="1">
          <animate attributeName="r" values="75;100;75" dur="15s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.03;0.1;0.03" dur="15s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g stroke="{c2}" stroke-width="0.9" fill="none" stroke-linecap="round">
        <path d="M 130,130 C 280,90 360,200 480,190 C 600,180 660,260 760,300 C 860,340 940,260 1080,250 C 1180,243 1240,290 1290,310">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="6.5s" repeatCount="indefinite"/>
        </path>
        <path d="M 130,130 C 200,250 150,360 250,420 C 350,480 320,560 380,650">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="7.2s" repeatCount="indefinite"/>
        </path>
        <path d="M 480,190 C 460,330 520,420 480,540 C 450,630 500,690 470,760">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="5.8s" repeatCount="indefinite"/>
        </path>
        <path d="M 760,300 C 700,420 780,470 740,580 C 710,660 760,700 730,770">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="6.9s" repeatCount="indefinite"/>
        </path>
        <path d="M 1080,250 C 1050,370 1110,430 1080,540 C 1055,630 1110,680 1090,760">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="6.1s" repeatCount="indefinite"/>
        </path>
        <path d="M 380,650 C 470,690 560,640 650,680 C 730,715 800,670 880,690">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="7.6s" repeatCount="indefinite"/>
        </path>
        <path d="M 880,690 C 950,660 1010,700 1090,760">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="6.3s" repeatCount="indefinite"/>
        </path>
        <path d="M 470,760 C 560,790 640,750 730,770">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="5.4s" repeatCount="indefinite"/>
        </path>
        <path d="M 250,420 C 350,380 420,440 480,540">
          <animate attributeName="stroke-opacity" values="{synapse_op_lo};{synapse_op_hi};{synapse_op_lo}" dur="6.7s" repeatCount="indefinite"/>
        </path>
      </g>

      <g filter="url(#softBlur)">
        <g stroke="{c2}" stroke-width="1.1" opacity="0.5">
          <line x1="130" y1="130" x2="95" y2="95"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="5s" repeatCount="indefinite"/></line>
          <line x1="130" y1="130" x2="100" y2="165"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="4.6s" repeatCount="indefinite"/></line>
          <line x1="130" y1="130" x2="170" y2="100"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="5.4s" repeatCount="indefinite"/></line>
        </g>
        <circle cx="130" cy="130" r="13" fill="url(#neuronCore)" opacity="{neuron_op}">
          <animate attributeName="r" values="11;15;11" dur="5.5s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <g stroke="{c1}" stroke-width="1.1" opacity="0.5">
          <line x1="480" y1="190" x2="445" y2="155"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="4.8s" repeatCount="indefinite"/></line>
          <line x1="480" y1="190" x2="450" y2="225"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="5.2s" repeatCount="indefinite"/></line>
        </g>
        <circle cx="480" cy="190" r="11" fill="url(#neuronCore2)" opacity="{neuron_op}">
          <animate attributeName="r" values="9;13;9" dur="6.1s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <g stroke="{c3}" stroke-width="1.1" opacity="0.5">
          <line x1="760" y1="300" x2="725" y2="265"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="5.6s" repeatCount="indefinite"/></line>
          <line x1="760" y1="300" x2="800" y2="270"><animate attributeName="opacity" values="0.3;0.55;0.3" dur="5s" repeatCount="indefinite"/></line>
        </g>
        <circle cx="760" cy="300" r="15" fill="url(#neuronCore3)" opacity="{neuron_op}">
          <animate attributeName="r" values="13;18;13" dur="5.8s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <circle cx="1080" cy="250" r="12" fill="url(#neuronCore)" opacity="{neuron_op}">
          <animate attributeName="r" values="10;14;10" dur="6.4s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <circle cx="380" cy="650" r="10" fill="url(#neuronCore2)" opacity="{neuron_op}">
          <animate attributeName="r" values="8;12;8" dur="5.3s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <circle cx="880" cy="690" r="11" fill="url(#neuronCore3)" opacity="{neuron_op}">
          <animate attributeName="r" values="9;13;9" dur="5.9s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <circle cx="470" cy="760" r="9" fill="url(#neuronCore)" opacity="{neuron_op}">
          <animate attributeName="r" values="7;11;7" dur="6.6s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <circle cx="1090" cy="760" r="10" fill="url(#neuronCore2)" opacity="{neuron_op}">
          <animate attributeName="r" values="8;12;8" dur="5.1s" repeatCount="indefinite"/>
        </circle>
      </g>

      <g filter="url(#softBlur)">
        <circle cx="250" cy="420" r="9" fill="url(#neuronCore3)" opacity="{neuron_op}">
          <animate attributeName="r" values="7;11;7" dur="6.8s" repeatCount="indefinite"/>
        </circle>
      </g>

      <circle r="2.8" fill="{c4}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="9s" repeatCount="indefinite"
          path="M 130,130 C 280,90 360,200 480,190 C 600,180 660,260 760,300 C 860,340 940,260 1080,250 C 1180,243 1240,290 1290,310"/>
        <animate attributeName="opacity" values="0;0.85;0.85;0" dur="9s" repeatCount="indefinite"/>
      </circle>

      <circle r="2.4" fill="{c2}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="11s" repeatCount="indefinite"
          path="M 130,130 C 200,250 150,360 250,420 C 350,480 320,560 380,650"/>
        <animate attributeName="opacity" values="0;0.8;0.8;0" dur="11s" repeatCount="indefinite"/>
      </circle>

      <circle r="2.4" fill="{c3}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="10s" repeatCount="indefinite"
          path="M 480,190 C 460,330 520,420 480,540 C 450,630 500,690 470,760"/>
        <animate attributeName="opacity" values="0;0.8;0.8;0" dur="10s" repeatCount="indefinite"/>
      </circle>

      <circle r="2.6" fill="{c1}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="12s" repeatCount="indefinite"
          path="M 760,300 C 700,420 780,470 740,580 C 710,660 760,700 730,770"/>
        <animate attributeName="opacity" values="0;0.8;0.8;0" dur="12s" repeatCount="indefinite"/>
      </circle>

      <circle r="2.4" fill="{c4}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="10.5s" repeatCount="indefinite"
          path="M 1080,250 C 1050,370 1110,430 1080,540 C 1055,630 1110,680 1090,760"/>
        <animate attributeName="opacity" values="0;0.8;0.8;0" dur="10.5s" repeatCount="indefinite"/>
      </circle>

      <circle r="2.2" fill="{c2}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="13s" repeatCount="indefinite"
          path="M 380,650 C 470,690 560,640 650,680 C 730,715 800,670 880,690"/>
        <animate attributeName="opacity" values="0;0.75;0.75;0" dur="13s" repeatCount="indefinite"/>
      </circle>

      <circle r="2.2" fill="{c3}" opacity="{particle_op}" filter="url(#particleBlur)">
        <animateMotion dur="9.5s" repeatCount="indefinite"
          path="M 470,760 C 560,790 640,750 730,770"/>
        <animate attributeName="opacity" values="0;0.75;0.75;0" dur="9.5s" repeatCount="indefinite"/>
      </circle>

      <g stroke="{c2}" stroke-width="1.3" fill="none" opacity="{wave_op}">
        <path d="M 0,820 C 60,800 90,840 150,820 C 210,800 240,845 300,820 C 360,800 390,840 450,820 C 510,800 540,845 600,820 C 660,800 690,840 750,820 C 810,800 840,845 900,820 C 960,800 990,840 1050,820 C 1110,800 1140,845 1200,820 C 1260,800 1290,840 1350,820 C 1380,810 1395,825 1400,820">
          <animate attributeName="d" dur="18s" repeatCount="indefinite"
            values="M 0,820 C 60,800 90,840 150,820 C 210,800 240,845 300,820 C 360,800 390,840 450,820 C 510,800 540,845 600,820 C 660,800 690,840 750,820 C 810,800 840,845 900,820 C 960,800 990,840 1050,820 C 1110,800 1140,845 1200,820 C 1260,800 1290,840 1350,820 C 1380,810 1395,825 1400,820;
                    M 0,820 C 60,840 90,800 150,820 C 210,840 240,795 300,820 C 360,840 390,800 450,820 C 510,840 540,795 600,820 C 660,840 690,800 750,820 C 810,840 840,795 900,820 C 960,840 990,800 1050,820 C 1110,840 1140,795 1200,820 C 1260,840 1290,800 1350,820 C 1380,830 1395,815 1400,820;
                    M 0,820 C 60,800 90,840 150,820 C 210,800 240,845 300,820 C 360,800 390,840 450,820 C 510,800 540,845 600,820 C 660,800 690,840 750,820 C 810,800 840,845 900,820 C 960,800 990,840 1050,820 C 1110,800 1140,845 1200,820 C 1260,800 1290,840 1350,820 C 1380,810 1395,825 1400,820"/>
        </path>
      </g>

      <g stroke="{c3}" stroke-width="1" fill="none" opacity="{wave_op}">
        <path d="M 0,865 C 70,850 100,880 170,865 C 240,850 270,882 340,865 C 410,850 440,882 510,865 C 580,850 610,882 680,865 C 750,850 780,882 850,865 C 920,850 950,882 1020,865 C 1090,850 1120,882 1190,865 C 1260,850 1290,882 1360,865 C 1380,860 1395,868 1400,865">
          <animate attributeName="d" dur="22s" repeatCount="indefinite"
            values="M 0,865 C 70,850 100,880 170,865 C 240,850 270,882 340,865 C 410,850 440,882 510,865 C 580,850 610,882 680,865 C 750,850 780,882 850,865 C 920,850 950,882 1020,865 C 1090,850 1120,882 1190,865 C 1260,850 1290,882 1360,865 C 1380,860 1395,868 1400,865;
                    M 0,865 C 70,880 100,850 170,865 C 240,880 270,848 340,865 C 410,880 440,848 510,865 C 580,880 610,848 680,865 C 750,880 780,848 850,865 C 920,880 950,848 1020,865 C 1090,880 1120,848 1190,865 C 1260,880 1290,848 1360,865 C 1380,872 1395,860 1400,865;
                    M 0,865 C 70,850 100,880 170,865 C 240,850 270,882 340,865 C 410,850 440,882 510,865 C 580,850 610,882 680,865 C 750,850 780,882 850,865 C 920,850 950,882 1020,865 C 1090,850 1120,882 1190,865 C 1260,850 1290,882 1360,865 C 1380,860 1395,868 1400,865"/>
        </path>
      </g>

      <g fill="{c4}" opacity="0.6">
        <circle cx="150" cy="500" r="1.8"><animate attributeName="cy" values="500;460;500" dur="13s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0.55;0" dur="13s" repeatCount="indefinite"/></circle>
        <circle cx="980" cy="120" r="1.6"><animate attributeName="cy" values="120;82;120" dur="11s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0.5;0" dur="11s" repeatCount="indefinite"/></circle>
        <circle cx="1250" cy="550" r="2"><animate attributeName="cy" values="550;505;550" dur="14s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0.55;0" dur="14s" repeatCount="indefinite"/></circle>
        <circle cx="620" cy="60" r="1.6"><animate attributeName="cy" values="60;25;60" dur="12s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0.5;0" dur="12s" repeatCount="indefinite"/></circle>
        <circle cx="60" cy="300" r="1.8"><animate attributeName="cy" values="300;260;300" dur="15s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0.5;0" dur="15s" repeatCount="indefinite"/></circle>
        <circle cx="1330" cy="780" r="1.6"><animate attributeName="cy" values="780;745;780" dur="13.5s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0.5;0" dur="13.5s" repeatCount="indefinite"/></circle>
      </g>
    </svg>
    """, unsafe_allow_html=True)


def render_topbar(show_toggle=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("""
        <div class="ns-logo"><span class="dot"></span> NeuroScan AI</div>
        """, unsafe_allow_html=True)
    with col2:
        if show_toggle:
            label = "🌙" if st.session_state.theme_mode == "dark" else "☀️"
            if st.button(label, key="theme_toggle_btn"):
                st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
                st.rerun()


def risk_band(final_risk):
    if final_risk >= 0.7:
        return "high", "🔴", "High Risk Detected", "#ef4444"
    elif final_risk >= 0.3:
        return "medium", "🟡", "Moderate Risk Detected", "#f59e0b"
    else:
        return "low", "🟢", "Low Risk", "#10b981"


def risk_recommendation(band):
    if band == "high":
        return "We strongly recommend scheduling a consultation with a neurologist as soon as possible for a thorough cognitive evaluation."
    elif band == "medium":
        return "Monitor memory changes closely, stay mentally active, and consider a cognitive screening with your doctor."
    else:
        return "Maintain a healthy lifestyle — regular exercise, quality sleep, balanced nutrition, and continued memory challenges."


def lifestyle_suggestions(band):
    if band == "high":
        return [
            "Schedule a neurologist consultation within the next 1-2 weeks",
            "Begin a structured daily routine to reduce confusion",
            "Arrange for a trusted family member to assist with appointments",
            "Avoid major financial or legal decisions until evaluated",
        ]
    elif band == "medium":
        return [
            "Engage in puzzles, reading or memory games daily",
            "Maintain consistent sleep and meal schedules",
            "Schedule a cognitive screening within the next 1-3 months",
            "Stay socially active and connected with family or friends",
        ]
    else:
        return [
            "Continue regular physical exercise (30 min, 5x/week)",
            "Maintain a balanced, heart-healthy diet",
            "Keep mentally active with new skills or hobbies",
            "Schedule routine check-ups annually",
        ]


def render_gauge_html(pct, color, dark_mode=True):
    track = "rgba(255,255,255,0.08)" if dark_mode else "rgba(37,99,235,0.12)"
    angle = max(0, min(360, int(pct * 3.6)))
    text_color = "#e8eefc" if dark_mode else "#0f172a"
    st.markdown(f"""
    <div class="gauge-wrap">
        <svg viewBox="0 0 220 220" width="220" height="220">
            <circle cx="110" cy="110" r="92" fill="none" stroke="{track}" stroke-width="16"/>
            <circle cx="110" cy="110" r="92" fill="none" stroke="{color}" stroke-width="16"
                stroke-linecap="round"
                stroke-dasharray="{(angle/360)*578:.1f} 578"
                transform="rotate(-90 110 110)">
                <animate attributeName="stroke-dasharray" from="0 578" to="{(angle/360)*578:.1f} 578" dur="1.1s" fill="freeze"/>
            </circle>
        </svg>
        <div class="gauge-center">
            <div class="gv" style="color:{color};">{pct}%</div>
            <div class="gl" style="color:{text_color};opacity:0.55;">Risk Index</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_bar_chart_html(mmse, speech_score, pct, dark_mode=True):
    mmse_h = int((mmse / 30) * 130) + 10
    speech_h = int((speech_score / 10) * 130) + 10
    pct_h = int((pct / 100) * 130) + 10
    st.markdown(f"""
    <div class="bar-chart-row">
        <div class="bar-chart-col">
            <div class="bar-chart-val">{mmse}</div>
            <div class="bar-chart-bar" style="height:{mmse_h}px; background:linear-gradient(180deg,#2563eb,#1d4ed8);"></div>
            <div class="bar-chart-lbl">MMSE /30</div>
        </div>
        <div class="bar-chart-col">
            <div class="bar-chart-val">{speech_score:.1f}</div>
            <div class="bar-chart-bar" style="height:{speech_h}px; background:linear-gradient(180deg,#06b6d4,#0891b2);"></div>
            <div class="bar-chart-lbl">Speech /10</div>
        </div>
        <div class="bar-chart-col">
            <div class="bar-chart-val">{pct}%</div>
            <div class="bar-chart-bar" style="height:{pct_h}px; background:linear-gradient(180deg,#7c8cf8,#4f5fce);"></div>
            <div class="bar-chart-lbl">Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_csv_report(patient, health, mmse, speech_score, pct, band):
    rows = [
        ("Patient Name", patient.get("name")),
        ("Age", patient.get("age")),
        ("Gender", patient.get("gender")),
        ("Date", time.strftime("%Y-%m-%d %H:%M")),
        ("Patient ID", patient.get("patient_id")),
        ("Smoking", health.get("smoking_label")),
        ("Alcohol Consumption", health.get("alcohol_label")),
        ("Physical Activity", health.get("physical_label")),
        ("Sleep Quality", health.get("sleep_label")),
        ("Blood Pressure", health.get("bp_label")),
        ("Family History", health.get("family_label")),
        ("Diabetes", health.get("diabetes_label")),
        ("Memory Complaints", health.get("memory_label")),
        ("MMSE Score", f"{mmse}/30"),
        ("Speech Score", f"{speech_score:.1f}/10"),
        ("Risk Percentage", f"{pct}%"),
        ("Risk Band", band.upper()),
    ]
    lines = ["Field,Value"]
    for k, v in rows:
        v_str = str(v).replace(",", ";")
        lines.append(f"{k},{v_str}")
    return "\n".join(lines).encode("utf-8")


def build_text_report(patient, health, mmse, speech_score, pct, band, recommendation, suggestions):
    lines = []
    lines.append("NEUROSCAN AI — ALZHEIMER'S RISK ASSESSMENT REPORT")
    lines.append("=" * 52)
    lines.append("")
    lines.append("PATIENT INFORMATION")
    lines.append("-" * 30)
    lines.append(f"Name        : {patient.get('name')}")
    lines.append(f"Age         : {patient.get('age')}")
    lines.append(f"Gender      : {patient.get('gender')}")
    lines.append(f"Date        : {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Patient ID  : {patient.get('patient_id')}")
    lines.append("")
    lines.append("RISK SUMMARY")
    lines.append("-" * 30)
    lines.append(f"MMSE Score       : {mmse} / 30")
    lines.append(f"Speech Score     : {speech_score:.1f} / 10")
    lines.append(f"Risk Percentage  : {pct}%")
    lines.append(f"Risk Level       : {band.upper()}")
    lines.append("")
    lines.append("HEALTH PROFILE")
    lines.append("-" * 30)
    lines.append(f"Smoking                      : {health.get('smoking_label')}")
    lines.append(f"Alcohol Consumption          : {health.get('alcohol_label')}")
    lines.append(f"Physical Activity            : {health.get('physical_label')}")
    lines.append(f"Sleep Quality                : {health.get('sleep_label')}")
    lines.append(f"Blood Pressure               : {health.get('bp_label')}")
    lines.append(f"Family History of Alzheimer's: {health.get('family_label')}")
    lines.append(f"Diabetes                     : {health.get('diabetes_label')}")
    lines.append(f"Memory Complaints            : {health.get('memory_label')}")
    lines.append("")
    lines.append("RECOMMENDATION")
    lines.append("-" * 30)
    lines.append(recommendation)
    lines.append("")
    lines.append("LIFESTYLE SUGGESTIONS")
    lines.append("-" * 30)
    for s in suggestions:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("-" * 52)
    lines.append("This report is generated by an AI screening tool for informational")
    lines.append("purposes only and does not constitute a medical diagnosis. Please")
    lines.append("consult a qualified neurologist or physician for professional evaluation.")
    return "\n".join(lines).encode("utf-8")


def render_progress(step_index):
    pills = ""
    for i in range(7):
        cls = "done" if i < step_index else ("active" if i == step_index else "")
        pills += f'<div class="step-pill {cls}"></div>'
    st.markdown(f"""
    <div class="step-label-row">
        <span class="step-name">{STEP_NAMES[step_index]}</span>
        <span class="step-counter">STEP {step_index + 1} / 7</span>
    </div>
    <div class="progress-steps">{pills}</div>
    """, unsafe_allow_html=True)


def goto(page_name, step_index):
    st.session_state.page = page_name
    st.session_state.step = step_index
    st.rerun()


def render_welcome_page():
    render_topbar()
    st.markdown("""
    <div class="hero-block">
        <span class="brain-icon">🧠</span>
        <div class="hero-title">NeuroScan<br><span>Alzheimer's Risk Prediction System</span></div>
        <p class="hero-subtitle">
            An AI-powered screening tool that combines health history, cognitive
            memory testing, and speech pattern analysis to help estimate Alzheimer's
            risk early — supporting timely conversations with medical professionals.
        </p>
    </div>
    <div class="ns-divider"></div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="feature-card"><span class="fi">🩺</span>
        <div class="ft">Health Profile</div>
        <div class="fd">Lifestyle &amp; medical history factors</div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card"><span class="fi">🧩</span>
        <div class="ft">Memory Test</div>
        <div class="fd">MMSE-style cognitive screening</div></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card"><span class="fi">🎙️</span>
        <div class="ft">Speech Analysis</div>
        <div class="fd">Whisper-powered voice assessment</div></div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="feature-card"><span class="fi">📊</span>
        <div class="ft">AI Risk Score</div>
        <div class="fd">Fusion model risk prediction</div></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)

    if st.button("Start Assessment →"):
        goto("patient_details", 1)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        **NeuroScan** is a research-oriented screening prototype that estimates Alzheimer's
        risk by combining three independent signals: structured health and lifestyle data,
        a short MMSE-style cognitive quiz, and speech-derived features extracted via
        Whisper transcription.

        **Technologies used:** Python, Streamlit, Joblib, Whisper

        **Dataset:** Structured clinical/lifestyle features (age, blood pressure, sleep,
        smoking, family history, etc.) paired with cognitive and speech-derived scores.
        """)

    st.markdown("""
    <p style="text-align:center; font-size:0.72rem; margin-top:1.2rem;" class="sub-desc">
        ⚠️ This tool is for screening and educational purposes only. It does not replace
        professional medical diagnosis. Always consult a qualified neurologist.
    </p>
    """, unsafe_allow_html=True)


def render_patient_details_page():
    render_topbar()
    render_progress(1)

    st.markdown("""
    <span class="sub-label">Before We Begin</span>
    <p class="section-title">Tell us a little about yourself</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        name = st.text_input("Full Name", placeholder="e.g. Ramesh Sharma", value=st.session_state.patient_name)
    with col2:
        age = st.number_input("Age", min_value=40, max_value=100, value=st.session_state.patient_age)

    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.patient_gender == "Male" else 1)
    concern = st.text_area("Any specific concerns? (optional)", placeholder="e.g. I've been forgetting names lately…",
                            value=st.session_state.patient_concern, height=85)

    st.markdown(f"""
    <div class="metric-row" style="margin-top:1rem; margin-bottom:1.2rem;">
        <div class="metric-chip"><div class="val">~8</div><div class="lbl">Minutes to complete</div></div>
        <div class="metric-chip"><div class="val">3</div><div class="lbl">Test stages</div></div>
        <div class="metric-chip"><div class="val">{st.session_state.patient_id}</div><div class="lbl">Patient ID</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back"):
            goto("welcome", 0)
    with c2:
        if st.button("Continue →"):
            if not name.strip():
                st.warning("Please enter your name to continue.")
            else:
                st.session_state.patient_name = name.strip()
                st.session_state.patient_age = age
                st.session_state.patient_gender = gender
                st.session_state.patient_concern = concern
                goto("health", 2)


def render_health_page():
    render_topbar()
    render_progress(2)

    st.markdown("""
    <span class="sub-label">Section 1 of 3</span>
    <p class="section-title">Health Profile</p>
    <p class="sub-desc">These lifestyle and clinical factors help the model assess baseline risk.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=40, max_value=100, value=st.session_state.patient_age, key="age_health")
    with col2:
        gender_sel = st.selectbox("Gender", ["Male", "Female"],
                                   index=0 if st.session_state.patient_gender == "Male" else 1, key="gender_health")
    gender = 0 if gender_sel == "Male" else 1

    col3, col4, col5 = st.columns(3)
    with col3:
        smoking_sel = st.selectbox("Smoking Habit", ["Never", "Occasionally", "Regularly"])
        smoking = 0 if smoking_sel == "Never" else 1
    with col4:
        alcohol_sel = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Frequently"])
        alcohol = 0 if alcohol_sel == "Never" else (5 if alcohol_sel == "Occasionally" else 8)
    with col5:
        physical_sel = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
        physical = 2 if physical_sel == "Low" else (5 if physical_sel == "Medium" else 8)

    col6, col7 = st.columns(2)
    with col6:
        sleep_sel = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
        sleep = 2 if sleep_sel == "Poor" else (5 if sleep_sel == "Average" else 8)
    with col7:
        bp_sel = st.selectbox("Blood Pressure", ["Normal", "High", "Don't Know"])
        if bp_sel == "Normal":
            hypertension, sysbp, diabp = 0, 120, 80
        elif bp_sel == "High":
            hypertension, sysbp, diabp = 1, 140, 90
        else:
            hypertension, sysbp, diabp = 0, 120, 80

    col8, col9, col10 = st.columns(3)
    with col8:
        family_sel = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
        family = 0 if family_sel == "No" else 1
    with col9:
        diabetes_sel = st.selectbox("Diabetes", ["No", "Yes"])
        diabetes = 0 if diabetes_sel == "No" else 1
    with col10:
        memory_sel = st.selectbox("Memory Complaints", ["No", "Yes"])
        memory = 0 if memory_sel == "No" else 1

    st.session_state.health_data = {
        "age": age, "gender": gender, "smoking": smoking, "alcohol": alcohol,
        "physical": physical, "sleep": sleep, "family": family, "diabetes": diabetes,
        "hypertension": hypertension, "sysbp": sysbp, "diabp": diabp, "memory": memory,
        "smoking_label": smoking_sel, "alcohol_label": alcohol_sel, "physical_label": physical_sel,
        "sleep_label": sleep_sel, "bp_label": bp_sel, "family_label": family_sel,
        "diabetes_label": diabetes_sel, "memory_label": memory_sel,
    }

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back"):
            goto("patient_details", 1)
    with c2:
        if st.button("Continue →"):
            goto("memory", 3)


def render_memory_page():
    render_topbar()
    render_progress(3)

    st.markdown("""
    <span class="sub-label">Section 2 of 3</span>
    <p class="section-title">Cognitive Memory Test</p>
    <p class="sub-desc">Answer all five questions as accurately as you can. Just do your best.</p>
    """, unsafe_allow_html=True)

    score = 0
    q1 = st.radio("1. What year is it?", ["2024", "2025", "2026", "2027"], horizontal=True)
    if q1 == "2026":
        score += 1

    q2 = st.radio("2. What month is it currently?", ["January", "March", "June", "December"], horizontal=True)
    if q2 == "June":
        score += 1

    q3 = st.radio("3. How many days are in a leap year?", ["364", "365", "366", "367"], horizontal=True)
    if q3 == "366":
        score += 1

    q4 = st.radio("4. If a clock shows 3:00 PM, what time after 5 hours?",
                   ["6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM"], horizontal=True)
    if q4 == "8:00 PM":
        score += 1

    q5 = st.radio("5. A farmer has 17 sheep, all but 9 die. How many are left?", ["8", "9", "17", "0"], horizontal=True)
    if q5 == "9":
        score += 1

    mmse = score * 6
    st.session_state.mmse_score = mmse

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.4rem;">
        <span style="font-size:0.8rem;" class="sub-desc">MMSE Score</span>
        <span class="score-badge">🧩 {mmse} / 30</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(mmse / 30)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back"):
            goto("health", 2)
    with c2:
        if st.button("Continue →"):
            goto("speech", 4)


def render_speech_page():
    render_topbar()
    render_progress(4)

    st.markdown("""
    <span class="sub-label">Section 3 of 3</span>
    <p class="section-title">Speech &amp; Recall Assessment</p>
    """, unsafe_allow_html=True)

    try:
        from speech_analysis import speech_to_text, introduction_score, recall_score, speech_remark
        from streamlit_mic_recorder import mic_recorder
        SPEECH_AVAILABLE = True
    except ImportError:
        SPEECH_AVAILABLE = False

    intro_score_value = st.session_state.get("intro_score_value", 0)
    recall_score_value = st.session_state.get("recall_score_value", 0)

    st.markdown("""
    <span class="sub-label">Speech Test 1 — Introduction</span>
    <p class="sub-desc">Introduce yourself: your name, age, where you live, and what you do daily.</p>
    """, unsafe_allow_html=True)

    if SPEECH_AVAILABLE:
        st.markdown("""
        <div class="mic-wrapper">
            <div class="mic-pulse-ring">
                <div class="ring"></div><div class="ring"></div><div class="ring"></div>
                <span class="mic-icon-static">🎙️</span>
            </div>
        """, unsafe_allow_html=True)
        intro_audio = mic_recorder(start_prompt="🎤 Record Introduction", stop_prompt="⏹ Stop Recording", key="intro_test")
        st.markdown('</div>', unsafe_allow_html=True)

        if intro_audio is not None:
            st.markdown("""
            <div class="waveform">
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            """, unsafe_allow_html=True)
            with open("intro.webm", "wb") as f:
                f.write(intro_audio["bytes"])
            intro_text = speech_to_text("intro.webm")
            st.markdown(f'<div class="transcript-bubble">"{intro_text}"</div>', unsafe_allow_html=True)
            intro_score_value = introduction_score(intro_text)
            st.session_state.intro_score_value = intro_score_value
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.4rem;margin-bottom:0.4rem;">
                <span class="sub-desc" style="margin:0;">Introduction Score</span>
                <span class="score-badge">🗣 {intro_score_value} / 10</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Speech recording requires the `streamlit_mic_recorder` and `speech_analysis` modules.")

    st.markdown("""
    <div class="ns-divider"></div>
    <span class="sub-label">Speech Test 2 — Word Recall</span>
    <p class="sub-desc">Study the five words below, hide them, then record yourself recalling as many as possible.</p>
    """, unsafe_allow_html=True)

    show_words = st.checkbox("Show memory words")
    if show_words:
        st.markdown("""
        <div class="word-badges">
            <span class="word-badge">APPLE</span>
            <span class="word-badge">CHAIR</span>
            <span class="word-badge">RIVER</span>
            <span class="word-badge">SCHOOL</span>
            <span class="word-badge">GARDEN</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <ol style="font-size:0.82rem; padding-left:1.2rem; line-height:2; margin:0.4rem 0 0.8rem;" class="sub-desc">
        <li>Memorise the words carefully.</li>
        <li>Uncheck "Show memory words" so they disappear.</li>
        <li>Close your eyes, then press Recall Words.</li>
        <li>Say all the words you remember aloud.</li>
    </ol>
    """, unsafe_allow_html=True)

    if SPEECH_AVAILABLE:
        st.markdown("""
        <div class="mic-wrapper">
            <div class="mic-pulse-ring">
                <div class="ring"></div><div class="ring"></div><div class="ring"></div>
                <span class="mic-icon-static">🎙️</span>
            </div>
        """, unsafe_allow_html=True)
        recall_audio = mic_recorder(start_prompt="🎤 Recall Words", stop_prompt="⏹ Stop Recording", key="recall_test")
        st.markdown('</div>', unsafe_allow_html=True)

        if recall_audio is not None:
            st.markdown("""
            <div class="waveform">
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            """, unsafe_allow_html=True)
            with open("recall.webm", "wb") as f:
                f.write(recall_audio["bytes"])
            recall_text = speech_to_text("recall.webm")
            st.markdown(f'<div class="transcript-bubble">"{recall_text}"</div>', unsafe_allow_html=True)
            recall_score_value = recall_score(recall_text)
            st.session_state.recall_score_value = recall_score_value
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.4rem;margin-bottom:0.4rem;">
                <span class="sub-desc" style="margin:0;">Recall Score</span>
                <span class="score-badge">🔁 {recall_score_value} / 10</span>
            </div>
            """, unsafe_allow_html=True)

    speech_score = (intro_score_value + recall_score_value) / 2
    st.session_state.speech_score = speech_score

    st.markdown(f"""
    <div class="ns-divider"></div>
    <div style="display:flex;align-items:center;gap:0.5rem; margin-bottom:0.4rem;">
        <span class="sub-desc" style="margin:0;">Overall Speech Score</span>
        <span class="score-badge">✨ {speech_score:.1f} / 10</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(speech_score / 10)

    if SPEECH_AVAILABLE and speech_score > 0:
        try:
            st.info(speech_remark(speech_score))
        except Exception:
            pass

    st.session_state.speech_available = SPEECH_AVAILABLE

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back"):
            goto("memory", 3)
    with c2:
        if st.button("🔍 Run Analysis →"):
            goto("processing", 5)


def render_processing_page():
    render_topbar(show_toggle=False)
    render_progress(5)

    placeholder = st.empty()

    steps = [
        "Scanning patient data...",
        "Analysing speech patterns...",
        "Evaluating memory test results...",
        "Running fusion prediction model...",
        "Generating risk score...",
        "Finalizing report...",
    ]

    health = st.session_state.get("health_data", {})
    mmse = st.session_state.get("mmse_score", 0)
    speech_score = st.session_state.get("speech_score", 0.0)
    speech_available = st.session_state.get("speech_available", False)

    for i, step_text in enumerate(steps):
        pct_done = int(((i + 1) / len(steps)) * 100)
        step_html = ""
        for j, s in enumerate(steps):
            if j < i:
                step_html += f'<div class="proc-step complete">✓ {s}</div>'
            elif j == i:
                step_html += f'<div class="proc-step active">⏳ {s}</div>'
            else:
                step_html += f'<div class="proc-step">{s}</div>'

        with placeholder.container():
            st.markdown(f"""
            <div class="processing-card">
                <span class="processing-brain">🧠</span>
                <div style="font-family:'Outfit',sans-serif; font-size:1.3rem; font-weight:700; margin-bottom:1rem;">
                    Processing Assessment
                </div>
                {step_html}
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct_done / 100)
        time.sleep(0.55)

    try:
        model = joblib.load("models/alzheimer_model.pkl")
        prediction = model.predict(pd.DataFrame([[
            health.get("age"), health.get("gender"), health.get("smoking"), health.get("alcohol"),
            health.get("physical"), health.get("sleep"), health.get("family"), health.get("diabetes"),
            health.get("hypertension"), health.get("sysbp"), health.get("diabp"), mmse, health.get("memory")
        ]], columns=["Age", "Gender", "Smoking", "AlcoholConsumption", "PhysicalActivity", "SleepQuality",
                     "FamilyHistoryAlzheimers", "Diabetes", "Hypertension", "SystolicBP", "DiastolicBP",
                     "MMSE", "MemoryComplaints"]))
        health_risk = int(prediction[0])
    except Exception:
        risk_factors = sum([
            1 if health.get("age", 0) > 70 else 0,
            health.get("family", 0), health.get("diabetes", 0), health.get("hypertension", 0),
            1 if health.get("smoking") == 1 else 0,
            1 if health.get("memory") == 1 else 0,
            1 if mmse < 18 else 0
        ])
        health_risk = min(1, risk_factors / 5)

    speech_risk = (10 - speech_score) / 10
    final_risk = health_risk * 0.7 + speech_risk * 0.3

    if mmse <= 6:
        final_risk = max(final_risk, 0.8)
    if speech_score <= 2 and speech_available:
        final_risk = max(final_risk, 0.8)

    st.session_state.final_risk = final_risk
    st.session_state.health_risk = health_risk

    record = {
        "name": st.session_state.patient_name,
        "age": st.session_state.patient_age,
        "patient_id": st.session_state.patient_id,
        "mmse": mmse,
        "speech_score": round(speech_score, 1),
        "risk_pct": round(final_risk * 100),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.history.append(record)

    goto("results", 6)


def render_results_page():
    render_topbar()
    render_progress(6)

    health = st.session_state.get("health_data", {})
    mmse = st.session_state.get("mmse_score", 0)
    speech_score = st.session_state.get("speech_score", 0.0)
    final_risk = st.session_state.get("final_risk", 0.0)
    pct = round(final_risk * 100)
    dark_mode = st.session_state.theme_mode == "dark"

    band, icon, heading, color = risk_band(final_risk)
    rec = risk_recommendation(band)
    suggestions = lifestyle_suggestions(band)

    st.markdown("""
    <span class="sub-label">Assessment Complete</span>
    <p class="section-title">Patient Report</p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="patient-card">
        <div style="font-family:'Outfit',sans-serif; font-size:1.2rem; font-weight:700;">
            {st.session_state.patient_name}
        </div>
        <div class="patient-grid">
            <div class="patient-field"><div class="pl">Age</div><div class="pv">{st.session_state.patient_age}</div></div>
            <div class="patient-field"><div class="pl">Gender</div><div class="pv">{st.session_state.patient_gender}</div></div>
            <div class="patient-field"><div class="pl">Date</div><div class="pv">{time.strftime("%Y-%m-%d")}</div></div>
            <div class="patient-field"><div class="pl">Patient ID</div><div class="pv">{st.session_state.patient_id}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if band == "high":
        st.markdown("""
        <div class="emergency-banner">
            <span style="font-size:1.6rem;">🚨</span>
            <div>
                <div class="et">High Risk — Please Seek Medical Attention Soon</div>
                <div class="ed">This result suggests a meaningful level of cognitive risk. We recommend contacting a neurologist promptly.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.2rem;">Risk Overview</p>', unsafe_allow_html=True)

    gcol, dcol = st.columns(2)
    with gcol:
        render_gauge_html(pct, color, dark_mode)
    with dcol:
        render_bar_chart_html(mmse, speech_score, pct, dark_mode)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip"><div class="val">{mmse}<span style="font-size:0.95rem;opacity:0.5;">/30</span></div><div class="lbl">MMSE Score</div></div>
        <div class="metric-chip"><div class="val">{speech_score:.1f}<span style="font-size:0.95rem;opacity:0.5;">/10</span></div><div class="lbl">Speech Score</div></div>
        <div class="metric-chip"><div class="val">{pct}<span style="font-size:0.95rem;opacity:0.5;">%</span></div><div class="lbl">Risk Index</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="risk-card {band}">
        <div style="font-size:1.8rem; margin-bottom:0.35rem;">{icon}</div>
        <div class="risk-title">{heading}</div>
        <div class="risk-rec">📋 <strong>Recommendation:</strong> {rec}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.2rem;">Lifestyle Suggestions</p>', unsafe_allow_html=True)
    for s in suggestions:
        st.markdown(f"""
        <div style="display:flex; gap:0.6rem; align-items:flex-start; margin-bottom:0.5rem;">
            <span style="color:{color}; font-weight:700;">→</span>
            <span class="sub-desc" style="margin:0;">{s}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.2rem;">Export Report</p>', unsafe_allow_html=True)

    patient_dict = {
        "name": st.session_state.patient_name,
        "age": st.session_state.patient_age,
        "gender": st.session_state.patient_gender,
        "patient_id": st.session_state.patient_id,
    }

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        txt_bytes = build_text_report(patient_dict, health, mmse, speech_score, pct, band, rec, suggestions)
        st.download_button("📄 Download Text Report", data=txt_bytes,
                            file_name=f"neuroscan_report_{st.session_state.patient_id}.txt",
                            mime="text/plain")
    with ecol2:
        csv_bytes = build_csv_report(patient_dict, health, mmse, speech_score, pct, band)
        st.download_button("📊 Download CSV Report", data=csv_bytes,
                            file_name=f"neuroscan_report_{st.session_state.patient_id}.csv",
                            mime="text/csv")

    st.markdown("""
    <p style="text-align:center;font-size:0.71rem;margin-top:1.2rem;" class="sub-desc">
        This screening is informational only. Always consult a qualified medical professional for diagnosis.
    </p>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        with st.expander("🕓 Patient History (this session)"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, width='stretch', hide_index=True)

    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reset Assessment"):
            for key in ["health_data", "mmse_score", "speech_score", "intro_score_value",
                        "recall_score_value", "final_risk", "health_risk", "speech_available"]:
                st.session_state.pop(key, None)
            goto("welcome", 0)
    with c2:
        if st.button("🏠 Back to Home"):
            goto("welcome", 0)


init_state()
inject_theme()
render_background_svg()

PAGE_ROUTER = {
    "welcome": render_welcome_page,
    "patient_details": render_patient_details_page,
    "health": render_health_page,
    "memory": render_memory_page,
    "speech": render_speech_page,
    "processing": render_processing_page,
    "results": render_results_page,
}

current_page = st.session_state.page
render_fn = PAGE_ROUTER.get(current_page, render_welcome_page)
render_fn()