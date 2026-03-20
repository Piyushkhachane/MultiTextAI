import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
from deep_translator import GoogleTranslator
from langdetect import detect

# ================== CONFIG ==================
st.set_page_config(page_title="MultiText AI", layout="wide")

NEWS_LABEL_MAP = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech"
}

# ================== HELPERS ==================

def safe_dataframe(results):
    if isinstance(results, dict):
        return pd.DataFrame([results])
    return pd.DataFrame(results)

def clean_results(df, value_col, threshold=1.0):
    df = df[df[value_col] > threshold]
    df = df.sort_values(by=value_col, ascending=False)
    return df

def plot_pie_chart(df, label_col, value_col, title):
    fig = px.pie(
        df,
        names=label_col,
        values=value_col,
        title=title,
        hole=0.5
    )

    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05]*len(df)
    )

    fig.update_layout(
        title_x=0.5,
        legend_title="Categories"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================== LOAD MODELS ==================

@st.cache_resource
def load_pipelines():
    sentiment_pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )
    toxic_pipe = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        return_all_scores=True
    )
    category_pipe = pipeline(
        "text-classification",
        model="textattack/distilbert-base-uncased-ag-news",
        return_all_scores=True
    )
    return sentiment_pipe, toxic_pipe, category_pipe

with st.spinner("🔄 Loading models..."):
    sentiment_pipe, toxic_pipe, category_pipe = load_pipelines()

# ================== UI ==================

st.markdown(
    "<h1 style='text-align: center;'>🤖 MultiText AI - Smart Text Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "**Supports Sentiment, Toxicity & News classification — with 🌐 Multilingual Translation!**"
)

task = st.selectbox(
    "📌 Choose a Task",
    ["Sentiment Analysis", "Toxic Comment Detection", "News Category Classification"]
)

text = st.text_area("✍️ Enter your text here (any language):", height=150)

# ================== MAIN ==================

if st.button("🔎 Analyze") and text.strip():

    # 🌐 Language Detection
    with st.spinner("🌐 Detecting language & translating..."):
        try:
            detected_lang = detect(text)
        except Exception as e:
            st.error(f"❌ Language detection error: {e}")
            st.stop()

        if detected_lang != "en":
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                st.markdown(f"🌐 **Detected Language**: `{detected_lang}` → English:\n> _{translated}_")
                text_to_use = translated
            except Exception as e:
                st.error(f"❌ Translation failed: {e}")
                st.stop()
        else:
            text_to_use = text
            st.success("✅ Input is already in English")

    # ⚙️ Model Execution
    with st.spinner("⚙️ Running model..."):

        # ================== SENTIMENT ==================
        if task == "Sentiment Analysis":
            results = sentiment_pipe(text_to_use)[0]
            df = safe_dataframe(results)

            df['score'] = (df['score'] * 100).round(2)
            df = df.rename(columns={'label': 'Sentiment', 'score': 'Confidence (%)'})

            df = clean_results(df, 'Confidence (%)')

            st.subheader("📊 Sentiment Results")
            st.dataframe(df, use_container_width=True)
            plot_pie_chart(df, 'Sentiment', 'Confidence (%)', "Sentiment Distribution")

        # ================== TOXICITY ==================
        elif task == "Toxic Comment Detection":
            results = toxic_pipe(text_to_use)[0]
            df = safe_dataframe(results)

            df['score'] = (df['score'] * 100).round(2)
            df = df.rename(columns={'label': 'Toxic Label', 'score': 'Confidence (%)'})

            # 🔥 Keep top 3 labels only (important)
            df = df.sort_values(by='Confidence (%)', ascending=False).head(3)

            st.subheader("☣️ Toxicity Results")
            st.dataframe(df, use_container_width=True)
            plot_pie_chart(df, 'Toxic Label', 'Confidence (%)', "Top Toxicity Signals")

        # ================== NEWS ==================
        elif task == "News Category Classification":
            results = category_pipe(text_to_use)[0]

            if isinstance(results, dict):
                results = [results]

            for r in results:
                r['label'] = NEWS_LABEL_MAP.get(r['label'], r['label'])
                r['score'] = round(r['score'] * 100, 2)

            df = pd.DataFrame(results)
            df = df.rename(columns={'label': 'Category', 'score': 'Confidence (%)'})

            df = df.sort_values(by='Confidence (%)', ascending=False)

            st.subheader("📰 News Category Results")
            st.dataframe(df, use_container_width=True)
            plot_pie_chart(df, 'Category', 'Confidence (%)', "Category Distribution")

else:
    st.info("👉 Enter text and click **Analyze** to see results.")
