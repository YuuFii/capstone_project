from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import src.data.fetch_youtube as fetch
import src.models.sentiment_analysis as sentiment
import src.data.preprocess as preprocess
import pandas as pd
from src.data.db import get_collection

app = FastAPI()

class CommentRequest(BaseModel):
    text: str

class VideoRequest(BaseModel):
    video_ids: List[str]
    api_key: str
    max_comments: int

@app.get("/")
def read_root():
    return {
        "message": "Youtube Comment Sentiment Analysis API is running"
    }

@app.post("/predict")
def predict_sentiment(request: CommentRequest):
    try:
        result = preprocess.clean_text(request.text)
        result = sentiment.analyze_text(result)

        return {
            "text": request.text,
            "label": result['label'],
            "confidence": result['confidence']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_video")
def analyze_youtube_video(request: VideoRequest):
    try:
        all_results = []

        video_ids = request.video_ids
        api_key = request.api_key
        max_comments = request.max_comments

        for video_id in video_ids:
            raw_path = f"data/{video_id}_comments.csv"
            cleaned_path = f"data/{video_id}_comments_cleaned.csv"
            result_path = f"data/results_{video_id}_comments.csv"

            # Step 1: Fetch comments
            fetch.get_comments_from_videos(
                video_ids=[video_id],
                api_key=api_key,
                output_file=raw_path,
                max_comments_per_video=max_comments
            )

            # Step 2: Preprocess comments
            preprocess.text_preprocessing(input_path=raw_path, output_path=cleaned_path)

            # Step 3: Sentiment analysis
            sentiment.analyze_sentiment(input_path=cleaned_path, output_path=result_path)

            # Step 4: Load results to return
            df = pd.read_csv(result_path)

            # Limit response size if needed
            comments = []
            for _, row in df.iterrows():
                comments.append({
                    "comment": row.get("text", ""),
                    "label": row.get("sentiment", ""),
                    "confidence": float(row.get("confidence", 0.0))
                })

            all_results.append({
                "video_id": video_id,
                "comments": comments
            })

        return {
            "results": all_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def save_sentiment_to_mongo(comments):
    collection = get_collection()

    payload = {
        "comments": [
            {
                "text": comment["comment"],
                "label": comment["label"],
                "confidence": comment["confidence"]
            } for comment in comments
        ]
    }

    result = collection.insert_one(payload)
    print(f"💾 Data berhasil disimpan ke MongoDB dengan ID: {result.inserted_id}")
