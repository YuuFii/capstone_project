import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import os

# Load data hasil prediksi
df = pd.read_csv('data/youtube_comments_id_with_sentiment.csv')

# Pastikan folder results ada
os.makedirs("results/visualizations", exist_ok=True)

# 1️ Pie Chart - Distribusi Sentimen
plt.figure(figsize=(8, 6))
sentiment_counts = df['sentiment'].value_counts()
colors = {"positive": "green", "neutral": "gray", "negative": "red"}
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=[colors[label] for label in sentiment_counts.index])
plt.title('Distribusi Sentimen Komentar YouTube')
plt.axis('equal')  # Agar lingkaran tidak oval
plt.savefig('results/visualizations/sentiment_distribution_pie.png')
plt.show()

# 2️ WordCloud - Kata yang Sering Muncul
text_all = ' '.join(df['clean_text'].astype(str).str.lower().tolist())
stopwords = set(STOPWORDS)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stopwords,
    colormap='viridis'
).generate(text_all)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud - Kata Paling Umum dalam Komentar")
plt.tight_layout(pad=0)
plt.savefig('results/visualizations/wordcloud_overall.png')
plt.show()

# 3️ WordCloud per Sentimen
for sentiment in df['sentiment'].unique():
    text_by_sentiment = ' '.join(df[df['sentiment'] == sentiment]['clean_text'].astype(str).str.lower().tolist())
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        colormap='viridis' if sentiment == 'positive' else ('coolwarm' if sentiment == 'negative' else 'gray')
    ).generate(text_by_sentiment)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud - {sentiment.capitalize()}")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f'results/visualizations/wordcloud_{sentiment}.png')
    plt.show()