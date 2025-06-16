import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Deteksi Spam", page_icon="📩")
st.title("📩 Aplikasi Deteksi Pesan Spam")

@st.cache_resource
def train_model():
    df = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label_num']

    vect = CountVectorizer()
    X_vect = vect.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Simpan model & vectorizer untuk caching/percepatan
    with open("model.pkl", "wb") as m:
        pickle.dump(model, m)
    with open("vectorizer.pkl", "wb") as v:
        pickle.dump(vect, v)

    return model, vect

model, vect = train_model()

st.markdown("Masukkan pesan teks di bawah ini untuk mengecek apakah itu **spam** atau **bukan spam**:")

msg = st.text_input("📨 Pesan Anda:")

if st.button("🔍 Deteksi"):
    if not msg.strip():
        st.warning("⚠️ Silakan masukkan pesan terlebih dahulu.")
    else:
        vect_msg = vect.transform([msg])
        pred = model.predict(vect_msg)[0]
        st.success("✅ Bukan Spam" if pred == 0 else "🚨 Pesan Spam!")
