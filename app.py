import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import sys, asyncio
from huggingface_hub import snapshot_download
import os

# Windows에서 asyncio 이벤트 루프 정책 설정
if sys.platform.startswith("win") and sys.version_info >= (3, 11):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 스타일 적용
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<style>
.stApp{background:#ffffff;color:#000000;}
.main-wrap{max-width:720px;margin:0 auto;}

input,textarea{
    background:#fafafa !important;
    color:#000 !important;
    border:1px solid #dddddd !important;
    border-radius:8px !important;
}
textarea{resize:none !important;}
input:focus,textarea:focus{border-color:#6b5bff !important;}
::placeholder{color:#888 !important;}

.stButton>button{
    background:#6b5bff;color:#fff;border:none;
    padding:.55em 1.4em;border-radius:8px;transition:.2s;font-weight:600;
}
.stButton>button:hover{
    background:#8776ff;transform:translateY(-1px);
    box-shadow:0 4px 12px rgba(107,91,255,.35);
}

.intro-text{font-size:0.85rem;margin-top:-8px;}

.badge{
    display:inline-block;padding:.3em .7em;border-radius:6px;
    font-weight:600;font-size:1.05rem;
}
.badge.real{background:#20c997;color:#fff;}
.badge.fake{background:#f03e3e;color:#fff;}

.field-header{
    font-size:1.3rem;
    font-weight:700;
    margin:1.2rem 0 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    token = st.secrets["HF_TOKEN"]
    st.write("🔄 모델 다운로드 시작 중...")
    local_path = snapshot_download(
        repo_id="mikieoo/fake-news-bert",
        cache_dir="./hf_models",
        revision="main",
        token=token
    )
    st.write("✅ 모델 다운로드 완료")
    tokenizer = BertTokenizer.from_pretrained(local_path)
    model = BertForSequenceClassification.from_pretrained(local_path)
    model.eval()
    return tokenizer, model

# 모델 로드
tokenizer, model = load_model()

# UI 구성
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown("<h1><i class='fa-solid fa-newspaper'></i>&nbsp; Fake&nbsp;News&nbsp;Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='intro-text'>뉴스 제목과 본문을 입력하면 "
    "<b>진짜 뉴스(REAL)</b>인지 <b>가짜 뉴스(FAKE)</b>인지 예측합니다.</p>",
    unsafe_allow_html=True
)

st.markdown("<div class='field-header'>✓&nbsp;&nbsp;&nbsp;기사 제목</div>", unsafe_allow_html=True)
title = st.text_input(label="기사 제목 입력", placeholder="예) NASA confirms …", label_visibility="collapsed")

st.markdown("<div class='field-header'>✓&nbsp;&nbsp;&nbsp;기사 본문</div>", unsafe_allow_html=True)
text = st.text_area(
    label="기사 본문 입력",
    height=260,
    placeholder="여기에 뉴스 기사 전문을 붙여 넣으세요…",
    label_visibility="collapsed"
)

if st.button("예측하기", key="predict"):
    if not title or not text:
        st.warning("제목과 본문을 모두 입력해 주세요.")
    else:
        encoded = tokenizer(title + " " + text,
                            truncation=True, padding=True,
                            max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=1)[0]
        fake_p, real_p = probs.tolist()

        is_fake = fake_p > real_p
        confidence = max(fake_p, real_p) * 100

        headline = "가짜 뉴스 기사입니다." if is_fake else "진짜 뉴스 기사입니다."
        color = "#f03e3e" if is_fake else "#20c997"

        st.subheader("예측 결과")
        st.markdown(
            f"<p style='font-size:1.1rem;"
            f"color:{color};font-weight:700'>"
            f"{confidence:.2f}% 확률로 {headline}</p>",
            unsafe_allow_html=True
        )

        bar_html = f"""
        <div style="width:100%;height:22px;
                    background:#e9ecef;border-radius:11px;overflow:hidden">
            <div style="width:{fake_p*100:.2f}%;
                        height:100%;background:#f03e3e;float:left"></div>
            <div style="width:{real_p*100:.2f}%;
                        height:100%;background:#20c997;float:left"></div>
        </div>
        <p style="font-size:0.8rem;margin-top:2px">
            <span style="color:#f03e3e;font-weight:600">FAKE&nbsp;{fake_p*100:.2f}%</span> ·
            <span style="color:#20c997;font-weight:600">REAL&nbsp;{real_p*100:.2f}%</span>
        </p>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
