import pandas as pd
import streamlit as st

from nlp_pipeline import analyze

st.set_page_config(page_title="ระบบวิเคราะห์โพสต์เตือนภัย/อุบัติเหตุ", page_icon="🚨", layout="wide")

st.title("🚨 ระบบวิเคราะห์โพสต์เตือนภัย / ข่าวอุบัติเหตุ")
st.caption(
    "สกัดสถานที่เกิดเหตุ, วันเวลา, ผู้บาดเจ็บ/เสียชีวิต และหน่วยงานช่วยเหลือ จากข้อความไทย/อังกฤษ "
    "ด้วยเทคนิค Regex Cleansing, Tokenization/Normalization, Topic Identification และ NER แบบ Rule-based"
)

SAMPLE_TEXT = (
    "ด่วน!! เกิดเหตุไฟไหม้อาคารพาณิชย์ย่านถนนสุขุมวิท กรุงเทพมหานคร เมื่อเวลา 21.30 น. "
    "มีผู้บาดเจ็บ 3 ราย เสียชีวิต 1 ราย หน่วยกู้ภัยมูลนิธิร่วมกตัญญูและหน่วยดับเพลิงเข้าช่วยเหลือแล้ว "
    "โทรแจ้งเหตุด่วนที่ 1669 อ่านต่อ https://example.com/news #ไฟไหม้กรุงเทพ"
)

mode = st.sidebar.radio("โหมดการใช้งาน", ["วิเคราะห์ข้อความเดี่ยว", "วิเคราะห์แบบ Batch (CSV)"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**เทคนิคที่ใช้**\n"
    "- Regex & Cleansing (ลบ URL/แฮชแท็ก, สกัดเบอร์ฉุกเฉิน)\n"
    "- Tokenization & Normalization (PyThaiNLP)\n"
    "- Topic Identification (keyword scoring)\n"
    "- NER แบบ Rule/Dictionary (สถานที่, เวลา, ผู้บาดเจ็บ, หน่วยงาน)"
)


def render_result(result):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("ข้อความหลังทำความสะอาด (Cleaned Text)")
        st.write(result["cleaned_text"])
        with st.expander("Noise ที่ตัดออก / สกัดออกมา"):
            st.json(result["noise_removed"])
        with st.expander("Tokens (หลังตัด Stopwords)"):
            st.write(result["tokens_no_stopwords"])
    with col2:
        st.metric("ภาษา", result["language"])
        st.metric("ระดับความรุนแรง", result["severity"])
        st.metric("หัวข้อหลัก (Topic)", result["topic"])

    st.subheader("สัดส่วนหัวข้อ (Topic scores)")
    st.bar_chart(pd.Series(result["topic_scores"]))

    st.subheader("Named Entities ที่สกัดได้")
    e = result["entities"]
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.markdown("**📍 สถานที่**")
    ec1.write(e["locations"] or "-")
    ec2.markdown("**🕒 วันเวลา**")
    ec2.write(e["datetime_mentions"] or "-")
    ec3.markdown("**🚑 ผู้บาดเจ็บ/เสียชีวิต**")
    ec3.write(e["casualties"])
    ec4.markdown("**🏢 หน่วยงานช่วยเหลือ**")
    ec4.write(e["agencies"] or "-")


if mode == "วิเคราะห์ข้อความเดี่ยว":
    text = st.text_area("วางข้อความโพสต์เตือนภัย/ข่าวอุบัติเหตุ", value="", height=150, placeholder=SAMPLE_TEXT)
    if st.button("ใช้ข้อความตัวอย่าง"):
        text = SAMPLE_TEXT
        st.session_state["sample"] = SAMPLE_TEXT
    if "sample" in st.session_state and not text:
        text = st.session_state["sample"]

    if st.button("🔍 วิเคราะห์", type="primary"):
        if text.strip():
            result = analyze(text)
            render_result(result)
        else:
            st.warning("กรุณาใส่ข้อความก่อนวิเคราะห์")

else:
    st.write("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ชื่อ `text` (ดูตัวอย่างใน sample_data.csv)")
    uploaded = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("ไฟล์ CSV ต้องมีคอลัมน์ชื่อ 'text'")
        else:
            results = [analyze(t) for t in df["text"].astype(str)]
            out = pd.DataFrame({
                "text": df["text"],
                "language": [r["language"] for r in results],
                "topic": [r["topic"] for r in results],
                "severity": [r["severity"] for r in results],
                "locations": [", ".join(r["entities"]["locations"]) for r in results],
                "datetime_mentions": [", ".join(r["entities"]["datetime_mentions"]) for r in results],
                "injured": [r["entities"]["casualties"]["injured"] for r in results],
                "dead": [r["entities"]["casualties"]["dead"] for r in results],
                "agencies": [", ".join(r["entities"]["agencies"]) for r in results],
            })
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "⬇️ ดาวน์โหลดผลลัพธ์ (CSV)",
                out.to_csv(index=False).encode("utf-8-sig"),
                file_name="analysis_result.csv",
                mime="text/csv",
            )
