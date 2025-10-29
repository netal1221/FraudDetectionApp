# ===========================================================
# 💳 CREDIT CARD FRAUD DETECTION — FINAL FIXED APP (Netal)
# ===========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# -----------------------------------------------------------
# ⚙️ PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# -----------------------------------------------------------
# 🌈 STYLING
# -----------------------------------------------------------
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.navbar {
    background-color: #002b5c;
    height: 58px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 35px;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.nav-btn {
    background-color: transparent;
    border: none;
    color: white;
    font-size: 17px;
    font-weight: 600;
    padding: 10px 18px;
    border-radius: 8px;
    transition: 0.3s;
    cursor: pointer;
}
.nav-btn:hover {background-color: #004b8d;}
.active {background-color: #004b8d !important;}
section.main > div:has(> div.block-container) {padding-top: 70px !important;}
.main-box {
    background-color: #fff;
    border-radius: 12px;
    padding: 35px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    animation: fadeIn 0.8s ease-in;
}
@keyframes fadeIn {from {opacity: 0; transform: translateY(15px);} to {opacity: 1; transform: translateY(0);}}
.stButton>button {
    background-color: #004b8d;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 16px;
    transition: 0.3s;
}
.stButton>button:hover {background-color: #003870; transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 📦 DOWNLOAD FILES FROM GOOGLE DRIVE (if missing)
# -----------------------------------------------------------
# ✅ Tere final Google Drive file IDs
drive_files = {
    "fraud_model.pkl": "1fnad3ZFIUvFYe3AeK4eYUPV06vLk26Rm",
    "scaler.pkl": "1RlOHjYxuBRNCOnCbtd4fkmsuJMf93ZyK",
    "creditcard.csv": "1LAADBVYqcf4Isjln32bN6_tXeYfluXHh"
}

for filename, file_id in drive_files.items():
    if not os.path.exists(filename):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)
            st.success(f"✅ Downloaded {filename} from Google Drive.")
        except Exception as e:
            st.warning(f"⚠️ Could not download {filename}: {e}")

# -----------------------------------------------------------
# 🧭 PAGE STATE (PERSIST AFTER RELOAD)
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = st.query_params.get("page", ["home"])[0]

def switch_page(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

page = st.session_state.page

# -----------------------------------------------------------
# 🧭 NAVIGATION BAR
# -----------------------------------------------------------
st.markdown('<div class="navbar">', unsafe_allow_html=True)
cols = st.columns(5)
pages = ["home", "data", "predict", "chat", "about"]
labels = ["🏠 Home", "📊 Dataset", "🚀 Predict", "🤖 Chatbot", "ℹ️ About"]
for i in range(5):
    if cols[i].button(labels[i], key=f"nav_{i}", use_container_width=False):
        switch_page(pages[i])
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 MODEL LOADING
# -----------------------------------------------------------
model_available = False
if os.path.exists("fraud_model.pkl") and os.path.exists("scaler.pkl"):
    try:
        model = joblib.load("fraud_model.pkl")
        scaler = joblib.load("scaler.pkl")
        model_available = True
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# -----------------------------------------------------------
# 🏠 HOME PAGE
# -----------------------------------------------------------
if page == "home":
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.title("💳 Credit Card Fraud Detection System")
    st.write("""
    ### 🎯 Objective
    Detect and prevent fraudulent credit card transactions using **Machine Learning**.

    ### ⚙️ Tech Stack
    - Python
    - Streamlit
    - Scikit-learn
    - Random Forest
    - Pandas, NumPy

    ### 🚀 How to Use
    👉 Go to the **Predict** page to test your own transactions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 📊 DATA PAGE
# -----------------------------------------------------------
elif page == "data":
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.header("📊 Dataset Information")
    try:
        df = pd.read_csv("creditcard.csv")
        st.dataframe(df.head(10))
        st.success(f"✅ Loaded successfully: {df.shape[0]} records, {df.shape[1]} features.")
        st.info("Feature Info:\n- V1–V28 = anonymized features\n- Amount = transaction amount\n- Class: 0 = Legitimate, 1 = Fraud")
    except Exception as e:
        st.error(f"⚠️ Could not load dataset: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 🚀 PREDICT PAGE
# -----------------------------------------------------------
elif page == "predict":
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.header("🚀 Predict Transaction Fraud")

    def build_features(selected, amount, time):
        v = np.zeros(28, dtype=float)
        for idx, val in selected.items():
            if 1 <= idx <= 28:
                v[idx - 1] = float(val)
        return np.concatenate((v, [float(amount), float(time)])).reshape(1, -1)

    def demo_predict(amount, v_sum, amt_th=2000, v_th=15, require_both=False):
        cond1, cond2 = amount >= amt_th, v_sum >= v_th
        return 1 if ((cond1 and cond2) if require_both else (cond1 or cond2)) else 0

    with st.form("form_predict"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0, step=10.0)
        with c2:
            time = st.number_input("⏱ Transaction Time (sec)", min_value=0.0, value=1000.0, step=1.0)
        st.divider()

        st.subheader("⚙️ Demo Settings")
        demo_enabled = st.checkbox("Enable Demo Override", value=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            amt_th = st.number_input("Fraud if Amount ≥", value=2000.0, step=100.0)
        with cc2:
            v_th = st.number_input("Fraud if Σ|V| ≥", value=15.0, step=1.0)
        with cc3:
            both = st.checkbox("Require both conditions", value=False)

        st.subheader("📈 Choose Features (V1–V28)")
        picked = st.multiselect("Select feature numbers", [str(i) for i in range(1, 29)], default=["1", "3", "5"])
        selected = {}
        for idx in picked[:5]:
            selected[int(idx)] = st.number_input(f"Value for V{idx}", value=0.0, format="%.4f", key=f"v_{idx}")

        submit = st.form_submit_button("🔍 Predict")
        if submit:
            features = build_features(selected, amount, time)
            v_sum = sum(abs(v) for v in selected.values())

            if demo_enabled:
                pred = demo_predict(amount, v_sum, amt_th, v_th, both)
                if pred == 1:
                    st.error("🚨 Fraudulent Transaction Detected (Demo Rule)!")
                else:
                    st.success("✅ Legitimate Transaction (Demo Rule).")
            elif model_available:
                scaled = scaler.transform(features)
                pred = model.predict(scaled)[0]
                if pred == 1:
                    st.error("🚨 Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Legitimate Transaction.")
            else:
                st.warning("⚠️ Model not found. Use demo mode or upload .pkl files.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 🤖 CHATBOT PAGE
# -----------------------------------------------------------
elif page == "chat":
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.header("🤖 Chatbot Assistant")
    q = st.text_input("💬 Ask about the project:")
    if q:
        q = q.lower()
        if "fraud" in q:
            st.info("Fraud detection uses ML to spot unusual transaction patterns.")
        elif "model" in q:
            st.info("We used Random Forest for accurate fraud prediction.")
        elif "dataset" in q:
            st.info("Dataset: 284,807 transactions, only 492 are fraudulent.")
        else:
            st.info("This project identifies suspicious credit card transactions.")
    st.markdown('</div>', unsafe_allow_html=True)
