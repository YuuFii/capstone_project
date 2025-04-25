import pandas as pd
import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType
import pyspark.sql.functions as F

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english') + stopwords.words('indonesian'))

# Inisialisasi Spark Session
spark = SparkSession.builder \
    .appName("YouTube Comments Analysis") \
    .getOrCreate()

# Baca data dengan Pandas (untuk contoh kecil) atau Spark (untuk data besar)
data_pd = pd.read_csv('youtube_comments.csv')
data_spark = spark.createDataFrame(data_pd)

# Fungsi untuk membersihkan teks (diadaptasi untuk PySpark UDF)
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    return ' '.join(tokens)

# Daftarkan UDF untuk PySpark
clean_text_udf = udf(clean_text, StringType())

# Preprocessing dengan Spark
data_spark = data_spark.withColumn("clean_text", clean_text_udf(col("text")))
data_spark = data_spark.withColumn("word_count", F.size(F.split(F.col("clean_text"), " ")))

# Filter komentar dengan minimal 5 kata
data_spark = data_spark.filter(col("word_count") > 5)

print("\nStatistik Panjang Teks (Jumlah Kata):")
data_spark.select("word_count").describe().show()

data_filtered_pd = data_spark.toPandas()

# Fungsi untuk analisis frekuensi kata (menggunakan Pandas/Spark)
def analyze_word_frequency(text_series, top_n=20):
    all_words = ' '.join(text_series).split()
    word_freq = Counter(all_words).most_common(top_n)
    
    freq_df = pd.DataFrame(word_freq, columns=['Kata', 'Frekuensi'])
    
    plt.figure(figsize=(10, 6))
    freq_df.sort_values(by='Frekuensi').plot.barh(x='Kata', y='Frekuensi', color='skyblue')
    plt.title(f'{top_n} Kata Paling Sering Muncul')
    plt.tight_layout()
    plt.show()
    
    return freq_df

# Analisis Frekuensi Kata
word_freq_df = analyze_word_frequency(data_filtered_pd['clean_text'])
print("\nFrekuensi Kata:")
print(word_freq_df)

# Generate WordCloud
all_text = ' '.join(data_filtered_pd['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud dari Komentar YouTube')
plt.show()

# Simpan hasil preprocessing (opsional)
data_spark.drop("word_count").write.csv("youtube_preprocessed_spark", mode="overwrite", header=True)

# Stop Spark session
spark.stop()
