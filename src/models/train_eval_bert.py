import os
import argparse
import mlflow
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and preprocess the dataset"""
    dataset = load_dataset('csv', data_files=file_path)
    return dataset

# fine-tune the pre-trained model


def fine_tune_model(model, train_dataset, val_dataset, epochs=3, batch_size=16):
    """Fine-tune the model on the training dataset"""
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return model

def evaluate_model(model, test_dataset):
    """Evaluate the model on the test dataset"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    predictions = model.predict(test_dataset)
    preds = np.argmax(predictions.logits, axis=1)
    labels = test_dataset['label']

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a BERT model for sentiment analysis.")
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file.')
    # parser.add_argument('--model_name', type=str, default=MODEL, help='Pre-trained model name.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation.')

    args = parser.parse_args()

    # Load data
    # dataset = load_data(args.data_path)
    dataset = load_data('data/youtube_comments_with_sentiment.csv')
    
    # Split dataset into train, validation, and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=config)

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(model, train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate the model
    metrics = evaluate_model(fine_tuned_model, test_dataset)
    
    # # Log metrics to MLflow
    # mlflow.log_metrics(metrics)
    # mlflow.log_param("training_time", end_train - start_train)
    # mlflow.log_param("model_name", MODEL)
    # mlflow.log_param("training_time", end_train - start_train)
    # mlflow.log_param("epochs", args.epochs)
    # mlflow.log_param("batch_size", args.batch_size)
    # mlflow.log_artifact('./results', artifact_path='model')
    # mlflow.log_artifact('./logs', artifact_path='logs')

    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()