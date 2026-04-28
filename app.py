import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
from scipy.sparse import hstack

# ================= LOAD MODEL =================
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🚨",
    layout="centered"
)

# ================= HEADER =================
st.markdown("""
# 🚨 Fake Job / Internship Detection System
### AI-powered Fraud Detection using Machine Learning
""")

st.markdown("---")

# ================= INPUT =================
st.subheader("📌 Enter Job Details")

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input("Job Title")

with col2:
    salary_input = st.selectbox("Salary Provided?", ["No", "Yes"])

job_desc = st.text_area("Job Description", height=150)

st.markdown("---")

# ================= PREDICTION =================
if st.button("🔍 Analyze Job"):

    if job_desc.strip() == "":
        st.warning("⚠️ Please enter job description")
    else:
        # -------- TEXT PREPROCESSING --------
        text = (job_title + " " + job_desc).lower()
        text = re.sub(r'[^\w\s]', '', text)

        # -------- TF-IDF --------
        text_vec = vectorizer.transform([text])

        # -------- FEATURES --------
        salary_flag = 1 if salary_input == "Yes" else 0
        desc_length = len(text)

        extra_features = np.array([[salary_flag, desc_length]])
        final_input = hstack((text_vec, extra_features))

        # -------- MODEL --------
        proba = model.predict_proba(final_input)
        fake_prob = proba[0][1]

        # -------- RULE SYSTEM --------
        suspicious_words = [
            "earn", "quick money", "no experience", "work from home",
            "instant payment", "limited slots", "apply now", "guaranteed"
        ]

        rule_score = sum([1 for word in suspicious_words if word in text])
        final_score = fake_prob + (0.1 * rule_score)

        # -------- REASONS --------
        reasons = []

        if "no experience" in text:
            reasons.append("No experience required")

        if "earn" in text or "quick money" in text:
            reasons.append("Unrealistic earning claims")

        if "work from home" in text:
            reasons.append("Work-from-home scam pattern")

        if "apply now" in text or "limited slots" in text:
            reasons.append("Urgency tactics used")

        if "instant payment" in text:
            reasons.append("Suspicious payment promise")

        if salary_flag == 0:
            reasons.append("No salary information")

        if desc_length < 100:
            reasons.append("Very short description")

        found_words = [word for word in suspicious_words if word in text]

        # ================= RESULT =================
        st.markdown("## 📊 Job Risk Analysis")

        # -------- RISK STATUS --------
        if final_score > 0.7:
            st.error("🔴 HIGH RISK — This job is likely FAKE")
        elif final_score > 0.4:
            st.warning("🟠 SUSPICIOUS — Verify carefully before applying")
        else:
            st.success("🟢 SAFE — This job appears legitimate")

        # -------- METRICS --------
        col1, col2 = st.columns(2)
        col1.metric("Fraud Probability", f"{fake_prob:.2f}")
        col2.metric("Risk Score", f"{final_score:.2f}")

        # -------- PROGRESS BAR --------
        st.markdown("### 📉 Risk Level")
        st.progress(int(final_score * 100))

        # -------- REASONS --------
        st.markdown("### ⚠️ Key Risk Indicators")

        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("✔ No major risk indicators detected")

        # ================= 📈 FIXED GRAPH =================
        st.markdown("### 📈 Risk Breakdown")
        st.caption("Higher bars indicate stronger fraud indicators")

        risk_weights = {}

        if "no experience" in text:
            risk_weights["No Experience"] = 2

        if "earn" in text or "quick money" in text:
            risk_weights["Unrealistic Earnings"] = 3

        if "work from home" in text:
            risk_weights["Work From Home Risk"] = 2

        if "apply now" in text or "limited slots" in text:
            risk_weights["Urgency Pressure"] = 2

        if "instant payment" in text:
            risk_weights["Payment Scam"] = 3

        if salary_flag == 0:
            risk_weights["No Salary Info"] = 1

        if desc_length < 100:
            risk_weights["Short Description"] = 1

        if risk_weights:
            df_chart = pd.DataFrame({
                "Factor": list(risk_weights.keys()),
                "Impact": list(risk_weights.values())
            })

            st.bar_chart(df_chart.set_index("Factor"))
        else:
            st.info("No significant risk factors detected")

        # -------- KEYWORDS --------
        st.markdown("### 🔍 Detected Risk Keywords")

        if found_words:
            st.write(", ".join(found_words))
        else:
            st.write("No suspicious keywords found")

        # -------- RECOMMENDATION --------
        st.markdown("### 💡 Recommendation")

        if final_score > 0.7:
            st.error("Avoid applying to this job. It is likely fraudulent.")
        elif final_score > 0.4:
            st.warning("Verify company details before applying.")
        else:
            st.success("Safe to proceed, but always verify job details.")

        # -------- EXPLANATION --------
        with st.expander("ℹ️ How the system works"):
            st.write("""
            - TF-IDF converts job text into numerical features  
            - Random Forest model detects fraud patterns  
            - Additional features:
                - Salary presence  
                - Description length  
            - Rule-based system detects scam keywords  
            """)