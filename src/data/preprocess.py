import pandas as pd
import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')

# stopwords english and indonesian
STOPWORDS = set(stopwords.words('english') + stopwords.words('indonesian'))

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = emoji.replace_emoji(text, replace=u'')  # Remove emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    text = ' '.join(tokens)  # Join tokens back to string
    return text

if __name__ == "__main__":
    data = pd.read_csv('data/youtube_comments.csv')
    data['clean_text'] = data['text'].astype(str).apply(clean_text)
    data['word_count'] = data['clean_text'].apply(lambda x: len(str(x).split()))
    
    data_filtered = data[data['word_count'] > 5].copy()
    data_filtered.reset_index(drop=True, inplace=True)
    data_filtered['lang'] = data_filtered['clean_text'].apply(lambda x: detect(x) if len(x) > 0 else 'unknown')
    data_filtered.drop(columns=['word_count'], inplace=True)
    
    data_filtered.to_csv('data/youtube_comments_cleaned.csv', index=False)
