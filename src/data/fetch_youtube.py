import os
import pandas as pd
from tqdm import tqdm
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    developerKey=API_KEY
)

def get_comments_from_video(youtube, video_id, all_comments, max_comments=None):
    """
    Fetch comments from a YouTube video using the YouTube Data API.
    
    Args:
        video_id (str): The ID of the YouTube video.
        max_comments (int): The maximum number of comments to fetch.
        
    Returns:
        list: A list of comments from the video.
    """
    try:
        # Call the YouTube API to get comments
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
        ).execute()

        # Loop through the response and extract comments
        while results:
            for item in results.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    "video_id": video_id,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author_display_name": comment["authorDisplayName"],
                    "text": comment["textDisplay"],
                    "published_at": comment["publishedAt"],
                    "like_count": comment["likeCount"],
                })

                if max_comments and len(all_comments) >= max_comments:
                    return all_comments
                
            # Check if there are more comments to fetch
            if "nextPageToken" in results:
                results = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    pageToken=results["nextPageToken"],
                    maxResults=100,
                ).execute()
            else:
                break

    except Exception as e:
        print(f"An error occurred while retrieving comments from video {video_id}: {e}")

    return all_comments

def get_comments_from_videos(video_ids: list, max_comments_per_video=None, output_file="data/youtube_comments.csv"):
    """
    Fetch comments from a list of YouTube videos and save them to a CSV.
    
    Args:
        video_ids (list): A list of YouTube video IDs.
        max_comments_per_video (int, optional): Max comments per video. If None, fetch all.
        output_file (str): Path to save the output CSV.
    """
    all_comments = []

    for video_id in tqdm(video_ids, desc="Fetching comments"):
        print(f"Fetching comments for video ID: {video_id}...")
        before_count = len(all_comments)
        all_comments = get_comments_from_video(
            youtube,
            video_id,
            all_comments,
            max_comments=(before_count + max_comments_per_video) if max_comments_per_video else None
        )
        after_count = len(all_comments)
        if after_count > before_count:
            print(f"Fetched {after_count - before_count} new comments for video ID: {video_id}")
        else:
            print(f"No new comments fetched for video ID: {video_id}")

    print(f"Total comments fetched from {len(video_ids)} videos: {len(all_comments)}")
    df = pd.DataFrame(all_comments)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Comments saved to {output_file}")
    
if __name__ == "__main__":
    # Example usage
    video_ids = [
        "ZWpGshjeKHk",
        "z9or_nYdBe8",
        "B-9kUM7nsUU",
    ]

    get_comments_from_videos(video_ids, max_comments_per_video=None, output_file="data/youtube_comments.csv")

