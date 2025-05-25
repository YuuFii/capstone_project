import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from prometheus_client import Histogram, Gauge, Counter
from src.data.db import get_collection

confidence_score = Gauge('sentiment_avg_confidence', 'Rata-rata confidence score')
positive_comments = Gauge('sentiment_positive_count', 'Jumlah komentar positif')
negative_comments = Gauge('sentiment_negative_count', 'Jumlah komentar negatif')
neutral_comments = Gauge('sentiment_neutral_count', 'Jumlah komentar netral')
analysis_counter = Counter('sentiment_analysis_total', 'Total analisis dilakukan')
latency_histogram = Histogram('sentiment_analysis_latency_seconds', 'Latensi analisis sentimen')

def calculate_model_metrics(df):
    df = pd.DataFrame(df)
    avg_confidence = df['confidence'].mean()
    sentiment_distribution = df['label'].value_counts(normalize=True).to_dict()

    return {
        'avg_confidence': avg_confidence,
        'sentiment_distribution': sentiment_distribution
    }

def measure_performance(func, *args, **kwargs):
    start_time = time.time()

    cpu_start = psutil.cpu_percent(interval=None)
    memory_start = psutil.virtual_memory().percent

    result = func(*args, **kwargs)

    end_time = time.time()

    cpu_end = psutil.cpu_percent(interval=None)
    memory_end = psutil.virtual_memory().percent

    duration = end_time - start_time
    throughput = len(result['comments']) / duration if "comments" in result else 0

    return {
        'latency': duration,
        'throughput': throughput,
        'cpu_usage': (cpu_start + cpu_end) / 2,
        'memory_usage': (memory_start + memory_end) / 2
    }, result

def log_system_metrics(model_metrics, perf_metrics):
    payload = {
        "timestamp": datetime.now(timezone.utc),
        "model": {
            "avg_confidence": model_metrics["avg_confidence"],
            "sentiment_distribution": model_metrics["sentiment_distribution"]
        },
        "performance": {
            "latency": perf_metrics["latency"],
            "throughput": perf_metrics["throughput"],
            "cpu_usage": perf_metrics["cpu_usage"],
            "memory_usage": perf_metrics["memory_usage"]
        }
    }

    collection = get_collection("system_metrics")
    collection.insert_one(payload)
    print("✅ Metrik berhasil disimpan ke MongoDB.")
