import sys
sys.path.append(".")

if __name__ == "__main__":
    import src.data.fetch_youtube as fetch
    import src.data.preprocess as preprocess
    import src.models.sentiment_analysis as sentiment

    # video_ids = [
    #     "ZWpGshjeKHk",
    #     "z9or_nYdBe8",
    #     "B-9kUM7nsUU",
    # ]

    # fetch.get_comments_from_videos(
    #     video_ids,
    #     max_comments_per_video=None,
    #     output_file="data/youtube_comments.csv"
    # )

    preprocess.text_preprocessing(
        input_path='data/youtube_comments_id.csv',
        output_path='data/youtube_comments_id_cleaned.csv'
    )

    # sentiment.analyze_sentiment(
    #     input_path='data/youtube_comments_id_cleaned.csv',
    #     output_path='data/youtube_comments_id_with_sentiment.csv'
    # )
    