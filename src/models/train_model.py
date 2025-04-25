import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset as HFDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('data/youtube_comments_with_sentiment.csv')
df = df[['clean_text', 'sentiment']].dropna()

label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
# mapping of labels to integers
label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
print("Label mapping:", label_mapping)

train_text, val_text, train_labels, val_labels = train_test_split(
    df['clean_text'].tolist(),
    df['sentiment'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

train_encodings = tokenize_function(train_text)
val_encodings = tokenize_function(val_text)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(label_mapping)
).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1': f1,
    }

training_args = TrainingArguments(
    output_dir='./models/bert_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained('./models/bert_model')
tokenizer.save_pretrained('./models/bert_model')

print("Model training complete and saved to './models/bert_model'")