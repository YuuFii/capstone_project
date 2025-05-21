import streamlit as st
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from nltk.corpus import stopwords
import nltk
import os

nltk.download('stopwords')

API_URL = os.getenv("API_URL", "http://localhost:8000")


st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("🎥 YouTube Sentiment Analyzer")

st.markdown("""
<style>
    .stTextInput input, .stTextArea textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Tabbed Interface
tab1, tab2, tab3 = st.tabs(["📥 Analisis Komentar dari Video", "✍️ Analisis Satu Kalimat", "📊 Visualisasi Hasil Analisis"])

# --- TAB 1: Ambil Komentar dari Video ---
with tab1:
    st.subheader("Masukkan Video ID dan YouTube API Key")
    video_ids_input = st.text_area("YouTube Video ID (pisahkan dengan koma jika lebih dari satu):")
    api_key = st.text_input("YouTube API Key:", type="password")
    max_comments = st.number_input("Jumlah maksimal komentar per video", step=50 ,value=200)
    st.warning("Perhatikan dalam penggunaan quota API!")

    help_link = "[Cara membuat YouTube API Key](https://developers.google.com/youtube/registering_an_application )"
    st.markdown(f"Belum punya API Key? {help_link}", unsafe_allow_html=True)

    if st.button("🔍 Ambil Komentar & Analisis"):
        if not video_ids_input or not api_key:
            st.error("Harus mengisi semua field: Video ID(s) dan API Key.")
        else:
            video_ids = [vid.strip() for vid in video_ids_input.split(",") if vid.strip()]
            if not video_ids:
                st.warning("Tidak ada Video ID valid yang dimasukkan.")
            else:
                with st.spinner("Sedang mengambil dan menganalisis komentar..."):
                    payload = {"video_ids": video_ids, "api_key": api_key, "max_comments": max_comments}
                    response = requests.post(f"{API_URL}/analyze_video", json=payload)
                    
                    if response.status_code == 200:
                        all_data = response.json()

                        all_comments = []
                        
                        for video_data in all_data['results']:
                            video_id = video_data['video_id']
                            comments = video_data['comments']

                            all_comments.extend(comments)
                            
                            st.success(f"Video `{video_id}` - Ditemukan {len(comments)} komentar.")

                            st.markdown(f"Sampel {min(len(comments), 20)} komentar hasil analisis")
                            for idx, comment in enumerate(comments[:20]):
                                st.markdown(f"**{idx + 1}.** {comment['comment']}")
                                st.markdown(f"> **Sentimen**: `{comment['label'].upper()}` | **Confidence**: `{comment['confidence']:.2f}`")
                                st.divider()

                        st.session_state['analysis_data'] = all_comments
                    else:
                        error_text = response.json().get('message', 'Gagal mengambil komentar.')
                        st.error(f"Gagal mengambil komentar: {error_text}")

# --- TAB 2: Analisis Satu Kalimat ---
with tab2:
    st.subheader("Analisis Satu Komentar")
    user_text = st.text_area("Masukkan komentar untuk analisis sentimen:", height=150)
    
    if st.button("🚀 Analisis Komentar"):
        if not user_text.strip():
            st.warning("Silakan masukkan teks komentar.")
        else:
            payload = {"text": user_text}
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Hasil Analisis:")
                st.markdown(f"> **Komentar:** {result['text']}")
                st.markdown(f"> **Sentimen:** `{result['label'].upper()}`")
                st.markdown(f"> **Confidence:** `{result['confidence']:.4f}`")
            else:
                st.error("Gagal menganalisis komentar.")

# --- TAB 3: Visualisasi Hasil Analisis ---
with tab3:
    st.subheader("📊 Visualisasi Hasil Analisis Sentimen")

    if 'analysis_data' not in st.session_state:
        st.warning("Belum ada data analisis. Lakukan analisis terlebih dahulu.")
    else:
        df = pd.DataFrame(st.session_state['analysis_data'])

        # Tampilkan summary
        st.markdown("### 📝 Ringkasan Kontekstual")
        summary_path = "results/summary.txt"

        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = f.read()
            st.info(summary)
        else:
            st.warning("Tidak ada ringkasan tersedia.")

        # Tampilkan sampel data
        st.markdown("### 🔍 Sampel Data")
        st.write(df.head())

        # Distribusi Sentimen (Pie Chart) di dalam kolom
        st.markdown("### 📈 Distribusi Sentimen")
        col1, col2 = st.columns([2, 2])  # Buat dua kolom untuk layout yang lebih rapi

        with col1:
            sentiment_counts = df['label'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))  # Ukuran figure diperkecil
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
                   colors=['green', 'gray', 'red'])
            ax.axis('equal')
            st.pyplot(fig, use_container_width=True)

        # Word Cloud di dalam kolom
        with col2:
            st.markdown("### ☁️ Word Cloud Komentar")
            text_all = ' '.join(df['comment'].astype(str).str.lower().tolist())
            stop_words = set(stopwords.words('indonesian'))

            wordcloud = WordCloud(width=400, height=200, background_color='white',
                                  stopwords=stop_words).generate(text_all)  # Ukuran diperkecil

            plt.figure(figsize=(6, 3))  # Figure lebih kecil
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt, use_container_width=True)

        # Filter komentar berdasarkan label sentimen
        st.markdown("### 🔎 Lihat Komentar Berdasarkan Sentimen")
        selected_sentiment = st.selectbox("Pilih Sentimen", df['label'].unique(), key="vis_sentiment")
        filtered_comments = df[df['label'] == selected_sentiment]['comment']

        for comment in filtered_comments[:10]:  # Tampilkan maksimal 10 komentar
            st.markdown(f"- {comment}")