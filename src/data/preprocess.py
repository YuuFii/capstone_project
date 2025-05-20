import pandas as pd
import re
import string
import emoji
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from indoNLP.preprocessing import replace_slang, replace_word_elongation

nltk.download('punkt_tab')
nltk.download('stopwords')

# stopwords english and indonesian
STOPWORDS = set(stopwords.words('english') + stopwords.words('indonesian'))

def clean_text(text: str):
    text = text.lower()
    text = unidecode(text).lower()  # Normalize unicode characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = emoji.replace_emoji(text, replace=u'')  # Remove emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = replace_slang(text)
    text = replace_word_elongation(text)

    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    text = ' '.join(tokens)  # Join tokens back to string
    return text


def text_preprocessing(input_path='data/youtube_comments_id.csv', output_path='data/youtube_comments_id_cleaned.csv'):
    data = pd.read_csv(input_path)
    data['clean_text'] = data['text'].astype(str).apply(clean_text)
    data['word_count'] = data['clean_text'].apply(lambda x: len(str(x).split()))

    data_filtered = data[data['word_count'] > 5].copy()
    data_filtered.reset_index(drop=True, inplace=True)

    spam_keywords = [
        'weton', 'alexis', 'lexs', 'lexis', 'hercules', 'aero', 'aeiao', 
        'axl', 'dewa', 'asia99', 'pawpaw', 'sgi', 'd o r a', 'dora',
        'h k', 'hoki', 'l e x', 'slot'
    ]

    spam_pattern = re.compile(
        '|'.join(spam_keywords), flags=re.IGNORECASE
    )

    data_filtered['is_spam'] = data_filtered['clean_text'].str.contains(
        spam_pattern, na=False
    )

    data_filtered['is_spam'] = data_filtered['is_spam'].astype(int)

    data_filtered = data_filtered[data_filtered['is_spam'] == 0].copy()
    data_filtered.drop(columns=['is_spam'], inplace=True)

    data_filtered.to_csv(output_path, index=False)
    print(f"[PREPROCESS] Data preprocessing completed. Cleaned data saved to {output_path}")

