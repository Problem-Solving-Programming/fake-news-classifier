import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib
import re
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
import nltk
from nltk.corpus import stopwords

# 한글 폰트 설정 (macOS용)
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="AI 뉴스 진위 판별기",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Nanum+Gothic&display=swap');
    html, body, [class*="css"] {
        font-family: 'Nanum Gothic', sans-serif;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        text-shadow: 0 0 8px #00ffc3, 0 0 20px #00ffc3;
    }
    label, .stTextInput label, .stTextArea label, .stMarkdown p {
        color: #ffffff !important;
    }
    .stButton>button {
        background: linear-gradient(to right, #00ffc3, #007bff);
        color: black;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        box-shadow: 0 0 15px #00ffc3;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #007bff, #00ffc3);
        transform: scale(1.08);
        box-shadow: 0 0 25px #00ffc3;
    }
    .stButton>button:focus:not(:active) {
        color: black !important;
    }
    .block-container {
        padding: 2rem;
        background-color: rgba(18, 23, 43, 0.95);
        border-radius: 20px;
        box-shadow: 0 0 25px rgba(0,255,195,0.4);
    }
    .stTextInput>div>div>input,
    .stTextArea>div>textarea {
        background-color: #1a1f35;
        color: #ffffff;
        border: 1px solid #00ffc3;
        border-radius: 10px;
        padding: 10px;
    }
    .real-box h4 {
        color: #00ff7f !important;
        text-shadow: 0 0 8px #00ff7f, 0 0 12px #00ff7f;
    }
    .fake-box h4 {
        color: #ff4d4d !important;
        text-shadow: 0 0 8px #ff4d4d, 0 0 12px #ff4d4d;
    }
    .buzz-box {
        background-color: rgba(255, 77, 77, 0.15);
        color: #ffcccc;
        padding: 10px;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 10px;
        text-shadow: 0 0 8px #ffaaaa;
    }
    .sentiment-box {
        background-color: rgba(0, 255, 195, 0.1);
        color: white;
        border: 2px solid #00ffc3;
        border-radius: 10px;
        padding: 10px 15px;
        margin-top: 10px;
        box-shadow: 0 0 12px #00ffc3;
        font-size: 17px;
        font-weight: 600;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #ffffff !important;
        opacity: 0.9;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown div, .stMarkdown h6, .stMarkdown strong,
    .stMarkdown em, .stMarkdown code {
        color: #ffffff !important;
    }
    [data-testid="stNotificationContentWarning"] {
        color: #ffdddd !important;
        background-color: rgba(255, 77, 77, 0.1) !important;
        border-left: 5px solid #ff4d4d;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
download_nltk()

st.markdown("<br><h2 style='color:white;'>🧪 뉴스 기사 가짜/진짜 판별기</h2>", unsafe_allow_html=True)
st.write("진위가 궁금한 뉴스 제목과 내용을 입력하면 AI가 판단해줘요!")

@st.cache_resource
def load_model():
    return joblib.load('fake_news_model.pkl')

model = load_model()

if "history" not in st.session_state:
    st.session_state["history"] = []

title = st.text_input("📰 뉴스 제목을 입력하세요")
content = st.text_area("📄 뉴스 본문을 입력하세요")

if st.button("✅ 판별하기"):
    if not content.strip():
        st.warning("❗ 뉴스 본문을 입력해주세요.")
    else:
        full_text = title + " " + content
        result_prob = model.predict_proba([full_text])[0][1]
        result_label = "FAKE" if result_prob > 0.5 else "REAL"
        result_percent = result_prob * 100 if result_label == "FAKE" else (1 - result_prob) * 100

        if result_label == "FAKE":
            st.markdown(f"<div style='background-color:#ffe6e6;padding:10px;border-radius:8px'><h4 style='color:#ff4d4d; text-shadow: 1px 1px 2px #cc0000;'>❌ 가짜 뉴스일 확률: {result_percent:.1f}%</h4></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#e6ffed;padding:10px;border-radius:8px'><h4 style='color:#00ff7f; text-shadow: 1px 1px 2px #007f4d;'>✅ 진짜 뉴스일 확률: {result_percent:.1f}%</h4></div>", unsafe_allow_html=True)

        try:
            sentiment = TextBlob(content).sentiment.polarity
            if sentiment > 0.2:
                sent_label = "긍정"
                sent_color = "#00ff7f"
            elif sentiment < -0.2:
                sent_label = "부정"
                sent_color = "#ff4d4d"
            else:
                sent_label = "중립"
                sent_color = "#ffa500"

            st.markdown(
                f"""
                <div class='sentiment-box' style='border-left: 8px solid {sent_color}; font-weight: 400;'>
                    📈 감정 점수: <span style='color:{sent_color}; font-size:18px; font-weight:400'>{sentiment:.2f}</span><br>
                    🧠 감정 결과: <span style='color:{sent_color}; font-size:18px; font-weight:400'>{sent_label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        except:
            sentiment = 0
            sent_label = "분석 실패"
            st.warning("⚠️ 감정 분석 중 오류 발생")

        buzzwords = ["shocking", "truth", "bombshell", "scandal", "alert"]
        contains_buzz = any(word in content.lower() for word in buzzwords)
        if contains_buzz:
            st.warning("🚨 자극적인 단어가 포함되어 있습니다!")

        if len(title.split()) < 4:
            st.warning("⚠️ 제목이 너무 짧습니다. (단어 수 4 미만)")
        if len(content.split()) < 30:
            st.warning("⚠️ 본문 내용이 짧아서 신뢰도가 낮을 수 있습니다.")

        st.session_state["history"].append({
            "제목": title,
            "결과": result_label,
            "확률(%)": round(result_percent, 1),
            "감정 점수": round(sentiment, 2),
            "감정 결과": sent_label,
            "자극단어포함": contains_buzz
        })

if st.session_state["history"]:
    st.markdown("### 📝 최근 예측 기록")
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df)
