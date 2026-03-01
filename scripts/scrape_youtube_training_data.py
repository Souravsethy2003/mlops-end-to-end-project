"""
Scrape YouTube comments from diverse videos for training data.
Uses the existing model to auto-label comments, saving to data/external/youtube_comments.csv.

Usage:
    python scripts/scrape_youtube_training_data.py
"""

import os
import sys
import csv
import time
import random
import logging

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Diverse set of popular YouTube videos across genres
VIDEO_URLS = [
    # Music
    "https://www.youtube.com/watch?v=JGwWNGJdvx8",  # Ed Sheeran
    "https://www.youtube.com/watch?v=RgKAFK5djSk",  # Wiz Khalifa
    "https://www.youtube.com/watch?v=kJQP7kiw5Fk",  # Luis Fonsi - Despacito
    "https://www.youtube.com/watch?v=60ItHLz5WEA",  # Alan Walker - Faded
    "https://www.youtube.com/watch?v=fRh_vgS2dFE",  # Justin Bieber - Sorry
    # Tech
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley (meme culture)
    "https://www.youtube.com/watch?v=8S0FDjFBj8o",  # MKBHD
    "https://www.youtube.com/watch?v=WXuK6gekU1Y",  # Linus Tech Tips
    # Education
    "https://www.youtube.com/watch?v=aircAruvnKk",  # 3Blue1Brown
    "https://www.youtube.com/watch?v=HVsySz-h9r4",  # Kurzgesagt
    # Gaming
    "https://www.youtube.com/watch?v=n_Dv4JMiwK8",  # PewDiePie
    "https://www.youtube.com/watch?v=ByC8sRdL-Ro",  # MrBeast
    # News / Commentary
    "https://www.youtube.com/watch?v=FG6HbWw2RF4",  # TED Talk
    "https://www.youtube.com/watch?v=arj7oStGLkU",  # TED Talk 2
    # Vlogs / Entertainment
    "https://www.youtube.com/watch?v=DLzxrzFCyOs",  # Popular vlog
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # First YouTube video
    # Sports
    "https://www.youtube.com/watch?v=FlsCjmMhFmw",  # Sports highlight
    # Science
    "https://www.youtube.com/watch?v=rWVAzS5duAs",  # Veritasium
    # Cooking
    "https://www.youtube.com/watch?v=OzRGGKDT_wE",  # Gordon Ramsay
    # Movie trailers
    "https://www.youtube.com/watch?v=TcMBFSGVi1c",  # Avengers
]

MAX_COMMENTS_PER_VIDEO = 500
OUTPUT_PATH = os.path.join(ROOT, 'data', 'external', 'youtube_comments.csv')


def scrape_comments(url, max_comments=MAX_COMMENTS_PER_VIDEO):
    """Scrape comments from a single YouTube video."""
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for c in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
            text = c.get('text', '').strip()
            if not text or len(text) < 10:
                continue
            comments.append(text)
            if len(comments) >= max_comments:
                break
    except Exception as e:
        logger.warning('Failed to scrape %s: %s', url, e)
    return comments


def auto_label(comments):
    """Use the existing LightGBM model to label comments."""
    import joblib
    import mlflow
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Load model
    mlflow.set_tracking_uri("http://localhost:5001/")
    from mlflow.tracking import MlflowClient
    model_uri = "models:/yt_chrome_plugin_model/1"
    model = mlflow.sklearn.load_model(model_uri)
    vectorizer = joblib.load(os.path.join(ROOT, 'tfidf_vectorizer.pkl'))

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = text.lower().strip()
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)
        text = ' '.join(w for w in text.split() if w not in stop_words)
        text = ' '.join(lemmatizer.lemmatize(w) for w in text.split())
        return text

    preprocessed = [preprocess(c) for c in comments]
    X = vectorizer.transform(preprocessed)
    predictions = model.predict(X).tolist()
    return [int(p) for p in predictions]


def main():
    all_comments = []
    all_labels = []

    # Resume from existing file if present
    existing = set()
    if os.path.exists(OUTPUT_PATH):
        import csv as _csv
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                existing.add(row['clean_comment'])
        logger.info('Found %d existing comments, will skip duplicates', len(existing))

    for i, url in enumerate(VIDEO_URLS):
        logger.info('[%d/%d] Scraping %s', i + 1, len(VIDEO_URLS), url)
        comments = scrape_comments(url)
        # Filter duplicates
        comments = [c for c in comments if c not in existing]
        if not comments:
            logger.info('  No new comments found, skipping')
            continue

        logger.info('  Got %d new comments, labeling...', len(comments))
        labels = auto_label(comments)
        all_comments.extend(comments)
        all_labels.extend(labels)
        existing.update(comments)

        # Be nice to YouTube
        time.sleep(random.uniform(1, 3))

    if not all_comments:
        logger.info('No new comments scraped')
        return

    # Append to CSV
    file_exists = os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0
    with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['clean_comment', 'category'])
        for comment, label in zip(all_comments, all_labels):
            writer.writerow([comment, label])

    logger.info('Saved %d new comments to %s', len(all_comments), OUTPUT_PATH)
    logger.info('Label distribution: %s', {
        -1: all_labels.count(-1),
        0: all_labels.count(0),
        1: all_labels.count(1)
    })


if __name__ == '__main__':
    main()
