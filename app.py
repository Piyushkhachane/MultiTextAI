import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go
from deep_translator import GoogleTranslator
from langdetect import detect

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MultiText AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root ── */
:root {
    --bg:       #060910;
    --surface:  #0d1117;
    --card:     #111827;
    --card2:    #162032;
    --border:   #1c2a3e;
    --border2:  #243447;
    --cyan:     #00f0ff;
    --purple:   #9f5cf7;
    --orange:   #ff6b35;
    --green:    #00e676;
    --red:      #ff3d5a;
    --text:     #f0f4ff;
    --muted:    #4a6080;
    --muted2:   #6b80a0;
    --radius:   14px;
    --radius-sm:8px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.main .block-container {
    padding: 0 2.5rem 5rem !important;
    max-width: 1280px;
}

/* =========================================
   SIDEBAR
   ========================================= */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.6rem 1.25rem 2rem !important;
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.2rem;
}
.sb-brand-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 12px rgba(0,240,255,0.3); }
    50%      { box-shadow: 0 0 24px rgba(159,92,247,0.5); }
}
.sb-brand-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.55rem;
    letter-spacing: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.sb-tagline {
    font-size: 0.7rem;
    color: var(--muted2);
    letter-spacing: 0.5px;
    margin-bottom: 1.8rem;
    padding-left: 46px;
}
.sb-nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.sb-task {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.75rem 0.9rem;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    margin-bottom: 0.35rem;
    transition: all 0.2s ease;
}
.sb-task.active.cyan-task   { border-left: 3px solid var(--cyan);   background: rgba(0,240,255,0.05);  border-color: rgba(0,240,255,0.15); }
.sb-task.active.orange-task { border-left: 3px solid var(--orange); background: rgba(255,107,53,0.05); border-color: rgba(255,107,53,0.15); }
.sb-task.active.purple-task { border-left: 3px solid var(--purple); background: rgba(159,92,247,0.05); border-color: rgba(159,92,247,0.15); }
.sb-task-icon { font-size: 1.1rem; width: 26px; text-align: center; flex-shrink: 0; }
.sb-task-name { font-size: 0.82rem; font-weight: 600; color: var(--text); line-height: 1.2; }
.sb-task-desc { font-size: 0.7rem; color: var(--muted2); margin-top: 1px; }
.sb-divider { height: 1px; background: var(--border); margin: 1.2rem 0; }
.sb-model-box {
    background: rgba(0,240,255,0.04);
    border: 1px solid rgba(0,240,255,0.12);
    border-radius: var(--radius-sm);
    padding: 0.75rem 0.9rem;
    margin-bottom: 1.2rem;
}
.sb-model-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 0.3rem; }
.sb-model-name  { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--cyan); word-break: break-all; line-height: 1.4; }
.sb-pill-row    { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.sb-pill { font-size: 0.67rem; font-weight: 600; padding: 3px 9px; border-radius: 99px; letter-spacing: 0.3px; }
.sb-pill.cyan   { background: rgba(0,240,255,0.1);  color: var(--cyan);   border: 1px solid rgba(0,240,255,0.2); }
.sb-pill.purple { background: rgba(159,92,247,0.1); color: var(--purple); border: 1px solid rgba(159,92,247,0.2); }
.sb-pill.orange { background: rgba(255,107,53,0.1); color: var(--orange); border: 1px solid rgba(255,107,53,0.2); }
.sb-pill.green  { background: rgba(0,230,118,0.1);  color: var(--green);  border: 1px solid rgba(0,230,118,0.2); }

/* =========================================
   HERO
   ========================================= */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #0a1628 0%, #0e0b20 40%, #080e1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 3.2rem 3rem 2.8rem;
    margin: 0 -2.5rem 2.5rem;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,240,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,240,255,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    animation: grid-drift 20s linear infinite;
}
@keyframes grid-drift {
    0%   { transform: translateY(0); }
    100% { transform: translateY(48px); }
}
.hero-blob1 {
    position: absolute; top: -80px; right: -60px;
    width: 340px; height: 340px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,240,255,0.14) 0%, transparent 65%);
    animation: blob-float 8s ease-in-out infinite;
}
.hero-blob2 {
    position: absolute; bottom: -60px; left: 120px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(159,92,247,0.12) 0%, transparent 65%);
    animation: blob-float 11s ease-in-out infinite reverse;
}
.hero-blob3 {
    position: absolute; top: 20px; left: -40px;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,107,53,0.07) 0%, transparent 65%);
    animation: blob-float 14s ease-in-out infinite;
}
@keyframes blob-float {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(15px,-20px) scale(1.05); }
    66%      { transform: translate(-10px,10px) scale(0.97); }
}
.hero-content { position: relative; z-index: 2; }
.hero-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,240,255,0.08); border: 1px solid rgba(0,240,255,0.18);
    border-radius: 99px; padding: 4px 14px 4px 8px;
    font-size: 0.72rem; font-weight: 600; color: var(--cyan);
    letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 1.1rem;
    animation: fade-up 0.6s ease both;
}
.hero-chip-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--cyan);
    animation: blink 1.5s ease-in-out infinite;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem; letter-spacing: 4px; line-height: 0.9;
    margin: 0 0 1rem;
    animation: fade-up 0.7s 0.1s ease both;
}
.hero-title-main {
    display: block;
    background: linear-gradient(90deg, #ffffff 0%, #c8d8ff 60%, #7fb3ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-title-accent {
    display: block;
    background: linear-gradient(90deg, var(--cyan) 0%, var(--purple) 60%, var(--orange) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 1rem; color: var(--muted2); max-width: 520px;
    line-height: 1.65; margin: 0 0 2rem; font-weight: 400;
    animation: fade-up 0.7s 0.2s ease both;
}
.hero-sub strong { color: var(--text); font-weight: 600; }
.hero-task-strip { display: flex; gap: 0.6rem; flex-wrap: wrap; animation: fade-up 0.7s 0.3s ease both; }
.hero-task-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; border: 1px solid;
    transition: all 0.2s;
}
.hero-task-chip.active-chip { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0,0,0,0.3); }
.htc-cyan   { color: var(--cyan);   background: rgba(0,240,255,0.08);  border-color: rgba(0,240,255,0.25); }
.htc-orange { color: var(--orange); background: rgba(255,107,53,0.08); border-color: rgba(255,107,53,0.25); }
.htc-purple { color: var(--purple); background: rgba(159,92,247,0.08); border-color: rgba(159,92,247,0.25); }
@keyframes fade-up { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }

/* =========================================
   STAT CARDS
   ========================================= */
.stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 2rem;
}
.stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.3rem 1.6rem;
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.stat-card:hover { transform: translateY(-2px); border-color: var(--border2); }
.stat-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0;
    height: 2px; border-radius: 0 0 var(--radius) var(--radius);
}
.sc-cyan::after   { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.sc-purple::after { background: linear-gradient(90deg, transparent, var(--purple), transparent); }
.sc-orange::after { background: linear-gradient(90deg, transparent, var(--orange), transparent); }
.stat-icon  { font-size: 1.5rem; margin-bottom: 0.6rem; display: block; }
.stat-val   { font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; letter-spacing: 2px; line-height: 1; margin-bottom: 0.25rem; }
.sc-cyan   .stat-val { color: var(--cyan); }
.sc-purple .stat-val { color: var(--purple); }
.sc-orange .stat-val { color: var(--orange); font-size: 1.6rem; }
.stat-lbl   { font-size: 0.73rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted2); }

/* =========================================
   SECTION HEADERS
   ========================================= */
.sec-header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.1rem; }
.sec-line   { flex: 1; height: 1px; background: var(--border); }
.sec-title  { font-family: 'Bebas Neue', sans-serif; font-size: 1.35rem; letter-spacing: 2.5px; color: var(--text); }
.sec-dot    { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sd-cyan    { background: var(--cyan);   box-shadow: 0 0 8px var(--cyan); }
.sd-orange  { background: var(--orange); box-shadow: 0 0 8px var(--orange); }
.sd-purple  { background: var(--purple); box-shadow: 0 0 8px var(--purple); }

/* =========================================
   INPUT PANEL
   ========================================= */
.input-wrap {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.6rem 1.6rem 0.8rem;
    margin-bottom: 0.6rem; transition: border-color 0.3s;
}
.input-wrap:focus-within { border-color: var(--border2); box-shadow: 0 0 0 3px rgba(0,240,255,0.05); }
.input-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--muted2); margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 6px;
}
.input-label-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--cyan); animation: blink 1.5s ease-in-out infinite; }

div[data-testid="stTextArea"] textarea {
    background: #070c14 !important; border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important; color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
    line-height: 1.65 !important; padding: 1rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0,240,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,240,255,0.07) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder { color: var(--muted) !important; font-style: italic; }

.input-hint { font-size: 0.72rem; color: var(--muted); margin-bottom: 1rem; }

/* ANALYZE BUTTON */
div[data-testid="stButton"] > button {
    background-image: linear-gradient(135deg, var(--cyan) 0%, #0091ea 50%, var(--purple) 100%) !important;
    color: #050912 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important; letter-spacing: 3px !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    padding: 0.75rem 2rem !important; cursor: pointer !important;
    width: 100% !important; position: relative !important; overflow: hidden !important;
    transition: transform 0.15s, box-shadow 0.25s !important;
    box-shadow: 0 4px 24px rgba(0,240,255,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,240,255,0.35) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(0,240,255,0.2) !important;
}

/* =========================================
   RESULT CARDS
   ========================================= */
.result-wrap { animation: fade-up 0.5s ease both; }
.pred-banner {
    border-radius: var(--radius); padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.pb-cyan   { background: rgba(0,240,255,0.07);  border: 1px solid rgba(0,240,255,0.2); }
.pb-red    { background: rgba(255,61,90,0.07);  border: 1px solid rgba(255,61,90,0.2); }
.pb-purple { background: rgba(159,92,247,0.07); border: 1px solid rgba(159,92,247,0.2); }
.pb-orange { background: rgba(255,107,53,0.07); border: 1px solid rgba(255,107,53,0.2); }
.pred-label-small { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.3rem; opacity: 0.7; }
.pred-label-big   { font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; letter-spacing: 2px; line-height: 1; }
.pb-cyan   .pred-label-big { color: var(--cyan); }
.pb-red    .pred-label-big { color: var(--red); }
.pb-purple .pred-label-big { color: var(--purple); }
.pb-orange .pred-label-big { color: var(--orange); }
.pred-score-num   { font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; letter-spacing: 1px; line-height: 1; }
.pb-cyan   .pred-score-num { color: var(--cyan); }
.pb-red    .pred-score-num { color: var(--red); }
.pb-purple .pred-score-num { color: var(--purple); }
.pb-orange .pred-score-num { color: var(--orange); }
.pred-score-label { font-size: 0.65rem; color: var(--muted2); text-transform: uppercase; letter-spacing: 1px; }

/* Confidence Bars */
.conf-bar-wrap  { margin-bottom: 1.2rem; }
.conf-bar-item  { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.75rem; }
.conf-bar-label { font-size: 0.8rem; font-weight: 600; color: var(--text); width: 150px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conf-bar-track { flex: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 99px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 99px; }
.conf-bar-pct   { font-family: 'DM Mono', monospace; font-size: 0.78rem; font-weight: 500; width: 52px; text-align: right; flex-shrink: 0; }

/* Translation box */
.trans-box {
    background: rgba(159,92,247,0.06); border: 1px solid rgba(159,92,247,0.18);
    border-radius: var(--radius-sm); padding: 0.9rem 1.1rem; margin-bottom: 1.2rem;
    font-size: 0.875rem; animation: fade-up 0.4s ease both;
}
.trans-box .tl { color: var(--muted2); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; }
.trans-box .tv { color: #c4b5fd; font-style: italic; }
.trans-box code { background: rgba(159,92,247,0.15); color: #a78bfa; padding: 1px 7px; border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 0.78rem; }

/* Info callout */
.info-callout {
    background: rgba(0,240,255,0.03); border: 1px dashed rgba(0,240,255,0.12);
    border-radius: var(--radius); padding: 2.5rem; text-align: center;
    color: var(--muted); font-size: 0.9rem;
}
.info-callout .ic-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem;
    letter-spacing: 2px; color: var(--muted2); margin-bottom: 0.4rem;
}

/* Misc */
div[data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important; overflow: hidden !important; }
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
summary { color: var(--muted2) !important; font-size: 0.83rem !important; font-weight: 600 !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
div[data-testid="stRadio"] label { font-size: 0.85rem !important; color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ================== CONSTANTS ==================
NEWS_LABEL_MAP = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech"
}

TASK_META = {
    "Sentiment Analysis": {
        "icon": "💬", "short": "Sentiment",
        "desc": "Detect positive or negative tone in text",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "dot": "sd-cyan", "pb": "pb-cyan", "hex": "#00f0ff",
        "chip": "htc-cyan", "sc": "sc-cyan", "sb": "cyan-task",
    },
    "Toxic Comment Detection": {
        "icon": "☣️", "short": "Toxicity",
        "desc": "Identify harmful or abusive language",
        "model": "unitary/toxic-bert",
        "dot": "sd-orange", "pb": "pb-orange", "hex": "#ff6b35",
        "chip": "htc-orange", "sc": "sc-orange", "sb": "orange-task",
    },
    "News Category Classification": {
        "icon": "📰", "short": "News",
        "desc": "Classify news: World, Sports, Business, Sci/Tech",
        "model": "textattack/distilbert-base-uncased-ag-news",
        "dot": "sd-purple", "pb": "pb-purple", "hex": "#9f5cf7",
        "chip": "htc-purple", "sc": "sc-purple", "sb": "purple-task",
    }
}

# ================== HELPERS ==================
def safe_dataframe(results):
    return pd.DataFrame([results] if isinstance(results, dict) else results)

def render_conf_bars(df, label_col, val_col, color_hex):
    html = '<div class="conf-bar-wrap">'
    for _, row in df.iterrows():
        pct = row[val_col]
        html += f"""
        <div class="conf-bar-item">
            <div class="conf-bar-label">{row[label_col]}</div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill" style="width:{pct}%; background: linear-gradient(90deg, {color_hex}88, {color_hex});"></div>
            </div>
            <div class="conf-bar-pct" style="color:{color_hex};">{pct}%</div>
        </div>"""
    html += '</div>'
    return html

def plot_radar(df, label_col, val_col, title, color):
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    fig = go.Figure(go.Scatterpolar(
        r=df[val_col].tolist() + [df[val_col].iloc[0]],
        theta=df[label_col].tolist() + [df[label_col].iloc[0]],
        fill='toself',
        fillcolor=f'rgba({r},{g},{b},0.12)',
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color='#4a6080', family='DM Sans'), x=0.5),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], color='#1c2a3e', gridcolor='#1c2a3e', tickfont=dict(size=9, color='#4a6080')),
            angularaxis=dict(color='#4a6080', gridcolor='#1c2a3e', tickfont=dict(size=10, color='#c0cce0')),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#f0f4ff'),
        margin=dict(l=30, r=30, t=40, b=20),
        height=260, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_donut(df, label_col, val_col, color_hex):
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    palette = [color_hex, f'rgba({r},{g},{b},0.4)', '#1c2a3e', '#243447']
    fig = go.Figure(go.Pie(
        labels=df[label_col], values=df[val_col],
        hole=0.68,
        marker=dict(colors=palette[:len(df)], line=dict(color='#060910', width=2)),
        textinfo='percent',
        textfont=dict(size=10, family='DM Sans'),
        hovertemplate='<b>%{label}</b><br>%{value}%<extra></extra>',
    ))
    fig.add_annotation(
        text=f"<b>{df[val_col].iloc[0]}%</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=color_hex, family='Bebas Neue'),
        xref='paper', yref='paper',
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0), height=200,
    )
    st.plotly_chart(fig, use_container_width=True)

def confidence_explanation():
    with st.expander("📘 How to read confidence scores"):
        st.markdown("""
        - Scores are the model's probability (0–100%) for each label.
        - **Higher %** = greater model certainty. Compare relative gaps between labels.
        - Multi-label models (like toxic-bert) can flag multiple signals simultaneously.
        - Scores may not sum to exactly 100% due to floating-point rounding.
        """)

# ================== LOAD MODELS ==================
@st.cache_resource
def load_pipelines():
    s = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
    t = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)
    c = pipeline("text-classification", model="textattack/distilbert-base-uncased-ag-news", return_all_scores=True)
    return s, t, c

with st.spinner("⚡ Warming up AI models…"):
    sentiment_pipe, toxic_pipe, category_pipe = load_pipelines()

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">🧠</div>
        <div class="sb-brand-name">MultiText</div>
    </div>
    <div class="sb-tagline">Smart Text Classification Suite</div>
    <div class="sb-nav-label">Select Task</div>
    """, unsafe_allow_html=True)

    task = st.radio(
        "task",
        options=list(TASK_META.keys()),
        label_visibility="collapsed",
        format_func=lambda x: f"{TASK_META[x]['icon']}  {x}"
    )
    meta = TASK_META[task]

    for t_name, m in TASK_META.items():
        active_cls = f"active {m['sb']}" if t_name == task else ""
        st.markdown(f"""
        <div class="sb-task {active_cls}">
            <div class="sb-task-icon">{m['icon']}</div>
            <div>
                <div class="sb-task-name">{t_name}</div>
                <div class="sb-task-desc">{m['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-divider"></div>
    <div class="sb-model-box">
        <div class="sb-model-label">Active Model</div>
        <div class="sb-model-name">{meta['model']}</div>
    </div>
    <div class="sb-nav-label">Features</div>
    <div class="sb-pill-row">
        <span class="sb-pill cyan">🌐 Multilingual</span>
        <span class="sb-pill green">⚡ Cached</span>
        <span class="sb-pill purple">📊 Charts</span>
        <span class="sb-pill orange">🔬 3 Models</span>
    </div>
    """, unsafe_allow_html=True)

# ================== MAIN CONTENT ==================

# ── Hero ──
st.markdown(f"""
<div class="hero">
    <div class="hero-blob1"></div>
    <div class="hero-blob2"></div>
    <div class="hero-blob3"></div>
    <div class="hero-content">
        <div class="hero-chip">
            <span class="hero-chip-dot"></span>
            AI-Powered · Live Inference · 100+ Languages
        </div>
        <div class="hero-title">
            <span class="hero-title-main">MULTI</span>
            <span class="hero-title-accent">TEXT AI</span>
        </div>
        <p class="hero-sub">
            Classify text using <strong>3 state-of-the-art models</strong> — sentiment, toxicity &amp; news category.
            Supports any language with auto-translation.
        </p>
        <div class="hero-task-strip">
            <span class="hero-task-chip htc-cyan {'active-chip' if task=='Sentiment Analysis' else ''}">💬 Sentiment</span>
            <span class="hero-task-chip htc-orange {'active-chip' if task=='Toxic Comment Detection' else ''}">☣️ Toxicity</span>
            <span class="hero-task-chip htc-purple {'active-chip' if task=='News Category Classification' else ''}">📰 News</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ──
st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card sc-cyan">
        <span class="stat-icon">🤖</span>
        <div class="stat-val">3</div>
        <div class="stat-lbl">Models Loaded</div>
    </div>
    <div class="stat-card sc-purple">
        <span class="stat-icon">🌐</span>
        <div class="stat-val">100+</div>
        <div class="stat-lbl">Languages</div>
    </div>
    <div class="stat-card sc-orange">
        <span class="stat-icon">{meta['icon']}</span>
        <div class="stat-val">{meta['short']}</div>
        <div class="stat-lbl">Active Task</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ──
st.markdown(f"""
<div class="sec-header">
    <span class="sec-dot {meta['dot']}"></span>
    <span class="sec-title">Input Text</span>
    <div class="sec-line"></div>
</div>
<div class="input-wrap">
    <div class="input-label">
        <span class="input-label-dot"></span>
        Enter text in any language
    </div>
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    "text_input",
    height=160,
    placeholder="Type or paste text here… English, Hindi, French, Arabic, Spanish and 100+ more.",
    label_visibility="collapsed",
)
st.markdown('<div class="input-hint">🌍 Language is auto-detected &amp; translated to English before classification</div>', unsafe_allow_html=True)

col_btn, col_sp = st.columns([1, 2])
with col_btn:
    analyze = st.button("⚡  ANALYZE NOW", use_container_width=True)

# ================== ANALYSIS ==================
if analyze and text.strip():

    with st.spinner("🌐 Detecting language…"):
        try:
            detected_lang = detect(text)
        except Exception as e:
            st.error(f"Language detection failed: {e}")
            st.stop()

        if detected_lang != "en":
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                st.markdown(f"""
                <div class="trans-box">
                    <div class="tl">🌐 Detected Language</div>
                    <div style="margin-bottom:0.4rem;">
                        <code>{detected_lang}</code>
                        <span style="color:#4a6080;"> → translated to English</span>
                    </div>
                    <div class="tv">"{translated}"</div>
                </div>
                """, unsafe_allow_html=True)
                text_to_use = translated
            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.stop()
        else:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(0,230,118,0.06);
                border:1px solid rgba(0,230,118,0.18);border-radius:6px;padding:6px 14px;
                font-size:0.82rem;color:#00e676;margin-bottom:1rem;font-weight:600;">
                ✅ English detected — no translation needed
            </div>
            """, unsafe_allow_html=True)
            text_to_use = text

    with st.spinner("⚙️ Running model inference…"):

        st.markdown(f"""
        <div class="sec-header" style="margin-top:0.5rem;">
            <span class="sec-dot {meta['dot']}"></span>
            <span class="sec-title">Results — {task}</span>
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── SENTIMENT ──
        if task == "Sentiment Analysis":
            results = sentiment_pipe(text_to_use)[0]
            df = safe_dataframe(results)
            df['score'] = (df['score'] * 100).round(2)
            df = df.rename(columns={'label': 'Sentiment', 'score': 'Confidence (%)'})
            df = df.sort_values('Confidence (%)', ascending=False)
            top = df.iloc[0]
            is_pos = top['Sentiment'].upper() == 'POSITIVE'
            pb_cls = "pb-cyan" if is_pos else "pb-red"
            icon = "😊" if is_pos else "😠"

            st.markdown(f"""
            <div class="result-wrap">
                <div class="pred-banner {pb_cls}">
                    <div>
                        <div class="pred-label-small">Top Prediction</div>
                        <div class="pred-label-big">{icon} {top['Sentiment']}</div>
                    </div>
                    <div style="text-align:center;">
                        <div class="pred-score-num">{top['Confidence (%)']}</div>
                        <div class="pred-score-label">% Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(render_conf_bars(df, 'Sentiment', 'Confidence (%)', meta['hex']), unsafe_allow_html=True)
            with col2:
                plot_radar(df, 'Sentiment', 'Confidence (%)', "Confidence Radar", meta['hex'])
            with col3:
                plot_donut(df, 'Sentiment', 'Confidence (%)', meta['hex'])

            st.dataframe(df, use_container_width=True, hide_index=True)
            confidence_explanation()

        # ── TOXICITY ──
        elif task == "Toxic Comment Detection":
            results = toxic_pipe(text_to_use)[0]
            df = safe_dataframe(results)
            df['score'] = (df['score'] * 100).round(2)
            df = df.rename(columns={'label': 'Toxic Label', 'score': 'Confidence (%)'})
            df = df.sort_values('Confidence (%)', ascending=False).head(3)
            top = df.iloc[0]
            is_toxic = top['Confidence (%)'] > 50
            pb_cls = "pb-red" if is_toxic else "pb-cyan"
            alert = "⚠️ TOXIC" if is_toxic else "✅ CLEAN"

            st.markdown(f"""
            <div class="result-wrap">
                <div class="pred-banner {pb_cls}">
                    <div>
                        <div class="pred-label-small">Primary Signal</div>
                        <div class="pred-label-big">{alert}</div>
                        <div style="font-size:0.8rem;color:var(--muted2);margin-top:4px;">{top['Toxic Label']}</div>
                    </div>
                    <div style="text-align:center;">
                        <div class="pred-score-num">{top['Confidence (%)']}</div>
                        <div class="pred-score-label">% Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(render_conf_bars(df, 'Toxic Label', 'Confidence (%)', meta['hex']), unsafe_allow_html=True)
            with col2:
                plot_radar(df, 'Toxic Label', 'Confidence (%)', "Toxicity Radar", meta['hex'])
            with col3:
                plot_donut(df, 'Toxic Label', 'Confidence (%)', meta['hex'])

            st.dataframe(df, use_container_width=True, hide_index=True)
            confidence_explanation()

        # ── NEWS ──
        elif task == "News Category Classification":
            results = category_pipe(text_to_use)[0]
            if isinstance(results, dict):
                results = [results]
            for r in results:
                r['label'] = NEWS_LABEL_MAP.get(r['label'], r['label'])
                r['score'] = round(r['score'] * 100, 2)
            df = pd.DataFrame(results)
            df = df.rename(columns={'label': 'Category', 'score': 'Confidence (%)'})
            df = df.sort_values('Confidence (%)', ascending=False)
            top = df.iloc[0]

            st.markdown(f"""
            <div class="result-wrap">
                <div class="pred-banner pb-purple">
                    <div>
                        <div class="pred-label-small">Top Category</div>
                        <div class="pred-label-big">🏆 {top['Category']}</div>
                    </div>
                    <div style="text-align:center;">
                        <div class="pred-score-num">{top['Confidence (%)']}</div>
                        <div class="pred-score-label">% Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(render_conf_bars(df, 'Category', 'Confidence (%)', meta['hex']), unsafe_allow_html=True)
            with col2:
                plot_radar(df, 'Category', 'Confidence (%)', "Category Radar", meta['hex'])
            with col3:
                plot_donut(df, 'Category', 'Confidence (%)', meta['hex'])

            st.dataframe(df, use_container_width=True, hide_index=True)
            confidence_explanation()

elif analyze and not text.strip():
    st.warning("⚠️  Please enter some text before clicking Analyze.")
else:
    st.markdown("""
    <div class="info-callout">
        <div class="ic-title">Ready to Analyze</div>
        Enter text above and hit <strong style="color:#f0f4ff;">⚡ ANALYZE NOW</strong>
        to see AI-powered classification with interactive charts.
    </div>
    """, unsafe_allow_html=True)
