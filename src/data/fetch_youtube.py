import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import requests
import time

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_comments_from_video(video_id, all_comments, max_comments=None):
    """
    Fetch comments from a YouTube video using the YouTube Data API.
    
    Args:
        video_id (str): The ID of the YouTube video.
        all_comments (list): A list to store all comments.
        max_comments (int): The maximum number of comments to fetch.
        
    Returns:
        list: A list of comments from the video.
    """
    try:
        base_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': 100,
            'textFormat': 'plainText',
            'key': API_KEY,
        }

        # Loop through the response and extract comments
        while True:
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Failed to retrieve data from video {video_id}: {response.json()}")
                break

            data = response.json()

            for item in data.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    'video_id': video_id,
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'author': comment.get('authorDisplayName', ''),
                    'text': comment.get('textDisplay', '').replace('\n', ' ').strip(),
                    'like_count': comment.get('likeCount', 0),
                    'published_at': comment.get('publishedAt', '')
                })

                if max_comments and len(all_comments) >= max_comments:
                    return all_comments
                
            # Pindah ke halaman selanjutnya jika ada
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
            params['pageToken'] = next_page_token
            time.sleep(0.1)  # Hindari rate limit

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
    VIDEO_IDS = [
        "ZWpGshjeKHk",
        "z9or_nYdBe8",
        "B-9kUM7nsUU",
    ]

    get_comments_from_videos(VIDEO_IDS, max_comments_per_video=None, output_file="data/youtube_comments.csv")