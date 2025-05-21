import mlflow.transformers
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re
from collections import Counter
import mlflow
import logging

logging.basicConfig(
    filename='logs/sentiment_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment('Sentiment Analysis Experiment')

pretrained_name = 'w11wo/indonesian-roberta-base-sentiment-classifier'
tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_name)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,  # Penting untuk teks panjang
    padding=True      # Untuk input dengan panjang berbeda
)

# Fungsi prediksi yang menangani error
def predict_sentiment(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return 'neutral'  # Default untuk teks kosong
        
        result = sentiment_pipeline(
            text,
            max_length=512,
            truncation=True,
            padding=True,
        )  # Batasi 512 token
        return {
            "label": result[0]['label'],
            "confidence": result[0]['score']
        }
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return {
            "label": "neutral",
            "confidence": 0.0
        }

def analyze_text(text: str):
    result = sentiment_pipeline(
        text,
        truncation=True,
        padding=True,
        max_length=512
    )

    return {
        "label": result[0]['label'],
        "confidence": result[0]['score']
    }

# 1. Konfigurasi Model
def analyze_sentiment(input_path='data/youtube_comments_id_cleaned.csv', output_path='data/youtube_comments_id_with_sentiment.csv'):
    print("[SENTIMENT] Starting sentiment analysis...")

    # 2. Load Data
    data = pd.read_csv(input_path)

    # 3. Inisialisasi Pipeline Sentimen
    try:
        # 4. Analisis Sentimen dengan Progress Bar
        tqdm.pandas(desc="Analyzing Sentiment")
        data['sentiment_result'] = data['clean_text'].progress_apply(predict_sentiment)

        data[['sentiment', 'confidence']] = data['sentiment_result'].apply(
            lambda x: pd.Series([x['label'], x['confidence']])
        ).fillna({'label': 'neutral', 'confidence': 0.0})

        data.drop(columns=['sentiment_result'], inplace=True)
        
        # # Save model
        # model_dir = "models/sentiment_model"
        # os.makedirs(model_dir, exist_ok=True)
        # sentiment_pipeline.save_pretrained(model_dir)

        # 5. Analisis Hasil
        sentiment_counts = data['sentiment'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution (%):")
        print(sentiment_counts)

        # Generate summary
        summary = generate_contextual_summary(data)

        # Simpan ke file
        summary_path = "results/summary.txt"
        os.makedirs("results", exist_ok=True)
        with open(summary_path, "w") as f:
            f.write(summary)
        
        # 6. Simpan Hasil
        data.to_csv(output_path, index=False)
        print(f"\n[SENTIMENT] Analysis completed. Results saved to: {output_path}")
        
        with mlflow.start_run(run_name="Indonesian Roberta Sentiment Analysis") as run:
            mlflow.log_param("model_name", pretrained_name)
            mlflow.log_param("input_path", input_path)
            
            for label, value in sentiment_counts.items():
                mlflow.log_metric(f"percentage_{label}", value)

            # summary = {
            #     "total_comments": len(data),
            #     "positive_comments": len(data[data['sentiment'] == 'positive']),
            #     "neutral_comments": len(data[data['sentiment'] == 'neutral']),
            #     "negative_comments": len(data[data['sentiment'] == 'negative'])
            # }

            # mlflow.log_dict(summary, "summary.json")

            mlflow.log_artifact(summary_path, artifact_path="summary")
            
            # log model to mlflow
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model,
                    "tokenizer": tokenizer
                },
                artifact_path="models/sentiment_model",
                pip_requirements=["torch==2.6.0+cpu", "mlflow"]
            )
            
            visualize_sentiment(data, output_dir="results/visualizations")

            for viz_file in os.listdir("results/visualizations"):
                if viz_file.endswith(".png"):
                    mlflow.log_artifact(os.path.join("results/visualizations", viz_file), artifact_path="visualizations")

            print("Hasil analisis sentimen disimpan di MLflow.")

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {str(e)}")
        print(f"An error occurred: {str(e)}")
        return None
    
def extract_top_words(text_series, n=5):
    all_text = ' '.join(text_series.astype(str).str.lower().tolist())
    all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)
    words = all_text.split()
    return [word for word, _ in Counter(words).most_common(n)]

def generate_contextual_summary(df):
    total_comments = len(df)
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100

    positive_pct = sentiment_counts.get('positive', 0)
    negative_pct = sentiment_counts.get('negative', 0)
    neutral_pct = sentiment_counts.get('neutral', 0)

    # Ekstrak topik/kata kunci
    top_words_positive = extract_top_words(df[df['sentiment'] == 'positive']['clean_text'], n=3)
    top_words_negative = extract_top_words(df[df['sentiment'] == 'negative']['clean_text'], n=3)
    top_words_all = extract_top_words(df['clean_text'], n=5)

    # Buat paragraf ringkasan
    summary = (
        f"Dari {total_comments} komentar yang dianalisis, "
        f"{positive_pct:.1f}% penonton memberikan respon positif, "
        f"{negative_pct:.1f}% negatif, dan "
        f"{neutral_pct:.1f}% netral. "
    )

    if len(top_words_positive) > 0:
        summary += (
            f"Banyak penonton menyukai video karena hal-hal seperti "
            f"`{top_words_positive[0]}` dan `{top_words_positive[1]}`. "
        )
    
    if len(top_words_negative) > 0:
        summary += (
            f"Namun, beberapa penonton menyampaikan kritik terkait "
            f"`{top_words_negative[0]}` dan `{top_words_negative[1]}`. "
        )

    summary += (
        f"Beberapa kata yang sering muncul dalam komentar antara lain: "
        f"`{'`, `'.join(top_words_all[:5])}`."
    )

    return summary

def visualize_sentiment(df, output_dir="results/visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Pie Chart - Distribusi Sentimen
    sentiment_counts = df['sentiment'].value_counts()
    colors = {"positive": "green", "neutral": "gray", "negative": "red"}
    
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=[colors[label] for label in sentiment_counts.index])
    plt.title('Distribusi Sentimen Komentar YouTube')
    plt.axis('equal')
    pie_chart_path = os.path.join(output_dir, "sentiment_distribution_pie.png")
    plt.savefig(pie_chart_path)
    plt.close()

    # 2️⃣ WordCloud - Keseluruhan
    all_text = ' '.join(df['clean_text'].astype(str).str.lower().tolist())
    generate_wordcloud(all_text, os.path.join(output_dir, "wordcloud_overall.png"), title="Word Cloud - Semua Komentar")

    # 3️⃣ WordCloud per Sentimen
    for sentiment in df['sentiment'].unique():
        text_by_sentiment = ' '.join(df[df['sentiment'] == sentiment]['clean_text'].astype(str).str.lower().tolist())
        generate_wordcloud(text_by_sentiment, os.path.join(output_dir, f"wordcloud_{sentiment}.png"), title=f"Word Cloud - {sentiment.capitalize()}")

    print("📊 Visualisasi selesai.")


def generate_wordcloud(text, save_path, title="Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Example usage
    analyze_sentiment(
        input_path='data/youtube_comments_id_cleaned.csv',
        output_path='data/youtube_comments_id_with_sentiment.csv'
    )
    # analyze_text("mari kita kawal qris jangan sampai dikuasai asing")