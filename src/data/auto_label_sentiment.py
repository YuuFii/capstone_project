import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(filepath)
        # Ensure clean_comment is string and handle missing values
        df['clean_text'] = df['clean_text'].astype(str).str.strip()
        df = df[df['clean_text'].str.len() > 1]  # Filter empty comments
        logger.info(f"Data loaded successfully. {len(df)} comments to process.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def initialize_sentiment_model():
    try:
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        
        # Load tokenizer explicitly
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline with explicit components
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Sentiment model initialized")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

def predict_sentiment_batch(texts, model, batch_size=32):
    """Predict sentiment for a batch of texts"""
    try:
        results = model(texts, batch_size=batch_size, truncation=True)
        return [result['label'].lower() for result in results]
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return ["unknown"] * len(texts)

def process_comments(df, model):
    """Process comments in batches with progress tracking"""
    comments = df['clean_text'].tolist()
    sentiments = []
    
    # Process in batches for efficiency
    with ThreadPoolExecutor() as executor:
        batches = [comments[i:i + 100] for i in range(0, len(comments), 100)]
        
        for batch in tqdm(batches, desc="Processing comments"):
            try:
                batch_results = predict_sentiment_batch(batch, model)
                sentiments.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                sentiments.extend(["unknown"] * len(batch))
    
    df['sentiment'] = sentiments
    return df

def analyze_sentiment_distribution(df):
    """Analyze and log sentiment distribution"""
    sentiment_dist = df['sentiment'].value_counts(normalize=True)
    logger.info("\nSentiment Distribution:\n" + str(sentiment_dist))

def save_results(df, output_path):
    """Save results with error handling"""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    input_path = "data/youtube_comments_cleaned.csv"
    output_path = "data/youtube_comments_with_sentiment.csv"
    
    try:
        # Load and prepare data
        df = load_data(input_path)
        
        # Initialize model
        sentiment_model = initialize_sentiment_model()
        
        # Process comments
        df = process_comments(df, sentiment_model)
        
        # Analyze results
        analyze_sentiment_distribution(df)
        
        # Save results
        save_results(df, output_path)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    import torch  # Import here to avoid early GPU memory allocation
    main()