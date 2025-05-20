import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Judul dashboard
st.title("📊 Analisis Sentimen Komentar YouTube")
st.markdown("Visualisasi hasil analisis menggunakan model zero-shot BERT")

# Load dataset hasil prediksi
@st.cache_data
def load_data():
    return pd.read_csv("data/youtube_comments_id_with_sentiment.csv")

df = load_data()

# Tampilkan sampel data
st.subheader("🔍 Sampel Data")
st.write(df.head())

# Distribusi Sentimen (Pie Chart)
st.subheader("📈 Distribusi Sentimen")
sentiment_counts = df['sentiment'].value_counts()
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
       colors=['green', 'gray', 'red'])
ax.axis('equal')  # Membuat lingkaran bulat
st.pyplot(fig)

# Word Cloud
st.subheader("☁️ Word Cloud Komentar")
text_all = ' '.join(df['clean_text'].astype(str).str.lower().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_all)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)

# Filter komentar berdasarkan label sentimen
st.subheader("🔎 Lihat Komentar Berdasarkan Sentimen")
selected_sentiment = st.selectbox("Pilih Sentimen", df['sentiment'].unique())
filtered_comments = df[df['sentiment'] == selected_sentiment]['clean_text']

for comment in filtered_comments[:10]:  # Tampilkan maksimal 10 komentar
    st.markdown(f"- {comment}")