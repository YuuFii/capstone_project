import mlflow.transformers
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import mlflow
import logging

logging.basicConfig(
    filename='logs/sentiment_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

mlflow.set_tracking_uri("http://localhost:5000")
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
        )

        data.drop(columns=['sentiment_result'], inplace=True)
        
        # # Save model
        # model_dir = "models/sentiment_model"
        # os.makedirs(model_dir, exist_ok=True)
        # sentiment_pipeline.save_pretrained(model_dir)

        # 5. Analisis Hasil
        sentiment_counts = data['sentiment'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution (%):")
        print(sentiment_counts)
        
        # 6. Simpan Hasil
        data.to_csv(output_path, index=False)
        print(f"\n[SENTIMENT] Analysis completed. Results saved to: {output_path}")
        
        with mlflow.start_run(run_name="Indonesian Roberta Sentiment Analysis") as run:
            mlflow.log_param("model_name", pretrained_name)
            mlflow.log_param("input_path", input_path)
            
            for label, value in sentiment_counts.items():
                mlflow.log_metric(f"percentage_{label}", value)

            summary = {
                "total_comments": len(data),
                "positive_comments": len(data[data['sentiment'] == 'positive']),
                "neutral_comments": len(data[data['sentiment'] == 'neutral']),
                "negative_comments": len(data[data['sentiment'] == 'negative'])
            }

            mlflow.log_dict(summary, "summary.json")
            
            # log model to mlflow
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model,
                    "tokenizer": tokenizer
                },
                artifact_path="models/sentiment_model"
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

def visualize_sentiment(df, output_dir="results/visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Pie Chart - Distribusi Sentimen
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {"positive": "green", "neutral": "gray", "negative": "red"}
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
    stopwords = set()
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
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