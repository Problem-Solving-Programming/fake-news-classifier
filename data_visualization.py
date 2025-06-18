import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

st.set_page_config(page_title="📊 뉴스 데이터 분석", layout="wide")

st.title("📊 가짜 뉴스 데이터 분석")

@st.cache_data
def load_data():
    df = pd.read_csv("merged_data_complete.csv")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label_text"] = df["label"].map({0: "REAL", 1: "FAKE"})
    if "title_word_count" not in df.columns:
        df["title_word_count"] = df["title"].fillna("").apply(lambda x: len(str(x).split()))
    if "text_word_count" not in df.columns:
        df["text_word_count"] = df["text"].fillna("").apply(lambda x: len(str(x).split()))
    if "sentiment_label" not in df.columns or df["sentiment_label"].isnull().sum() > 0:
        def get_sentiment_label(score):
            if pd.isna(score): return None
            if score > 0.2:
                return "긍정"
            elif score < -0.2:
                return "부정"
            else:
                return "중립"
        df["sentiment_label"] = df["sentiment_score"].apply(get_sentiment_label)
    return df

df = load_data()

analysis_option = st.selectbox("분석 항목을 선택하세요", [
    "제목 단어 수 분포",
    "본문 단어 수 분포",
    "감정 점수 분포",
    "감정 결과 분포",
    "자극 단어 포함 비율",
    "시기별 FAKE 뉴스 트렌드",
    "카테고리별 FAKE 뉴스 비율",
    "FAKE/REAL 워드클라우드"
])

if analysis_option == "제목 단어 수 분포":
    st.subheader("✍️ 제목 단어 수 분포")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="label_text", y="title_word_count", ax=ax)
    st.pyplot(fig)

elif analysis_option == "본문 단어 수 분포":
    st.subheader("📄 본문 단어 수 분포")
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="label_text", y="text_word_count", ax=ax)
    st.pyplot(fig)

elif analysis_option == "감정 점수 분포":
    st.subheader("📉 감정 점수 분포")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x="sentiment_score", hue="label_text", fill=True, ax=ax)
    st.pyplot(fig)

elif analysis_option == "감정 결과 분포":
    st.subheader("😐 감정 결과 분포")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="sentiment_label", hue="label_text", ax=ax)
    st.pyplot(fig)

elif analysis_option == "자극 단어 포함 비율":
    st.subheader("🚨 자극적인 단어 포함 비율")
    buzzwords = ["shocking", "scandal", "explosive", "truth", "bombshell"]
    def has_buzz(text):
        text = str(text).lower()
        return any(word in text for word in buzzwords)
    df["buzz"] = df["text"].apply(has_buzz)
    buzz_ratio = df.groupby("label_text")["buzz"].mean()
    fig, ax = plt.subplots()
    ax.pie(buzz_ratio, labels=buzz_ratio.index, autopct='%1.1f%%', startangle=140)
    st.pyplot(fig)

elif analysis_option == "시기별 FAKE 뉴스 트렌드":
    st.subheader("📅 시기별 FAKE 뉴스 생성 추이")
    df_f = df[df["label_text"] == "FAKE"].dropna(subset=["date"]).copy()
    df_f["month"] = df_f["date"].dt.to_period("M").astype(str)
    trend = df_f["month"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=trend.index, y=trend.values, marker="o", ax=ax)
    st.pyplot(fig)

elif analysis_option == "카테고리별 FAKE 뉴스 비율":
    st.subheader("🗂 카테고리별 FAKE 뉴스 비율")
    if "subject" in df.columns:
        ratio = df.groupby("subject")["label"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=ratio.values, y=ratio.index, ax=ax, palette="coolwarm")
        st.pyplot(fig)
    else:
        st.warning("⚠️ subject 컬럼이 없습니다.")

elif analysis_option == "FAKE/REAL 워드클라우드":
    st.subheader("☁️ FAKE vs REAL 워드클라우드")
    stop_words = set(stopwords.words('english'))
    fake_text = ' '.join(df[df['label_text'] == "FAKE"]["text"].dropna())
    real_text = ' '.join(df[df['label_text'] == "REAL"]["text"].dropna())

    fake_wc = WordCloud(width=600, height=400, background_color='white', stopwords=stop_words).generate(fake_text)
    real_wc = WordCloud(width=600, height=400, background_color='white', stopwords=stop_words).generate(real_text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**FAKE 뉴스**")
        st.image(fake_wc.to_array())
    with col2:
        st.markdown("**REAL 뉴스**")
        st.image(real_wc.to_array())
