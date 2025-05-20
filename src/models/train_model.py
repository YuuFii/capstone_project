import os
import time
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from accelerate import Accelerator

# Setup accelerator
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

# Set MLflow tracking URI (opsional, lokal saja)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # ubah jika perlu
mlflow.set_experiment("sentiment_analysis_youtube")

# Load dataset
df = pd.read_csv('data/youtube_comments_with_sentiment.csv')
df = df[['clean_text', 'sentiment']].dropna()

# Encode labels
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
print("Label mapping:", label_mapping)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['clean_text'].tolist(),
    df['sentiment'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

def tokenize_function(texts):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment-latest'
)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./models/bert_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",  # Jangan ke WandB/Hub
    gradient_accumulation_steps=2,
    fp16=True,         # Mixed precision training (Accelerate will handle it)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# MLflow run
with mlflow.start_run(run_name="bert_finetune_youtube_comments"):
    start_train = time.time()
    trainer.train()
    end_train = time.time()

    # Save the model and tokenizer
    output_dir = './models/bert_model'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Logging artifacts to MLflow
    mlflow.log_param("model_name", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    mlflow.log_param("epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)

    # Evaluate
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    # Log training time
    mlflow.log_metric("training_time_seconds", end_train - start_train)

    # Log model
    mlflow.pytorch.log_model(model, "model")

print("Training complete.")
print(f"Training time: {end_train - start_train:.2f} seconds")
