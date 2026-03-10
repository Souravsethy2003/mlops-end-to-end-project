# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT
import requests as http_requests
import random
import uuid
import os
import json
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://localhost:5001/")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "/home/ubuntu/Mlflow/tfidf_vectorizer.pkl")


# model = joblib.load("/app/lgbm_model.pkl")
# vectorizer = joblib.load("/app/tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    raw_items = [{'text': c, 'votes': 0, 'timestamp': None} for c in comments]
    try:
        results = _predict(raw_items)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify(results)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

from langdetect import detect as _langdetect, LangDetectException

# Toxicity keyword tiers: (pattern, score_weight)
_TOXIC_TIERS = [
    (r'\bkys\b|kill your ?self', 10),
    (r'\b(die|murder|threat)\b', 8),
    (r'\b(hate you|piece of .{0,10}(trash|garbage)|worthless|disgusting piece)\b', 6),
    (r'\b(idiot|moron|imbecile|pathetic|worthless|loser|scum)\b', 4),
    (r'\b(stupid|dumb|trash|garbage|horrible|awful|terrible)\b', 2),
    (r'\b(hate|worst|useless|lame)\b', 1),
]

def _toxicity_score(text):
    score = 0
    t = text.lower()
    for pattern, weight in _TOXIC_TIERS:
        score += len(re.findall(pattern, t)) * weight
    return min(10, score)


def _detect_lang(text):
    if len(text) < 15:
        return 'unknown'
    try:
        return _langdetect(text)
    except LangDetectException:
        return 'unknown'


def _parse_votes(raw):
    """Normalize vote counts from various string formats to int."""
    try:
        if isinstance(raw, (int, float)):
            return int(raw)
        if not raw:
            return 0
        s = str(raw).replace(',', '').strip().upper()
        if s.endswith('K'):
            return int(float(s[:-1]) * 1_000)
        if s.endswith('M'):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith('B'):
            return int(float(s[:-1]) * 1_000_000_000)
        return int(float(s)) if s else 0
    except (ValueError, TypeError):
        return 0


def _fetch_comments(url, sort_flag, fetch_limit):
    """Fetch comments with votes and timestamps.
    fetch_limit=None means fetch every comment (generator exhausted naturally).
    """
    downloader = YoutubeCommentDownloader()
    items = []
    for c in downloader.get_comments_from_url(url, sort_by=sort_flag):
        text = c.get('text', '').strip()
        if not text:
            continue
        votes = _parse_votes(c.get('votes', 0))
        time_parsed = c.get('time_parsed')
        timestamp = None
        if time_parsed and hasattr(time_parsed, 'isoformat'):
            try:
                timestamp = time_parsed.isoformat()
            except Exception:
                pass
        items.append({'text': text, 'votes': votes, 'timestamp': timestamp})
        if fetch_limit is not None and len(items) >= fetch_limit:
            break
    return items


def _spam_score(text):
    """Heuristic spam/low-quality detection. Returns (is_spam, reason)."""
    words = text.split()
    # Too short
    if len(words) < 4:
        return True, 'short'
    # All caps
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.70:
        return True, 'caps'
    # URL present
    if re.search(r'https?://|www\.', text, re.IGNORECASE):
        return True, 'url'
    # Word repetition (single word > 40% of all words)
    if words:
        freq = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        if max(freq.values()) / len(words) > 0.40:
            return True, 'repetition'
    # Emoji-only (strip emojis and whitespace, nothing left)
    stripped = re.sub(r'[\U00010000-\U0010ffff\U00002600-\U000027BF\s]', '', text, flags=re.UNICODE)
    if not stripped:
        return True, 'emoji-only'
    return False, ''


def _extract_topics(results, n=8):
    """Extract top TF-IDF keywords per sentiment group. Returns dict of lists."""
    import numpy as np
    groups = {'1': [], '0': [], '-1': []}
    for r in results:
        s = r.get('sentiment', '0')
        groups[s].append(r.get('comment', ''))
    topics = {}
    label_map = {'1': 'positive', '0': 'neutral', '-1': 'negative'}
    feature_names = vectorizer.get_feature_names_out()
    for key, texts in groups.items():
        label = label_map[key]
        if not texts:
            topics[label] = []
            continue
        preprocessed = [preprocess_comment(t) for t in texts]
        X = vectorizer.transform(preprocessed)
        # Mean TF-IDF score across documents in this group
        mean_scores = np.asarray(X.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[::-1][:n]
        keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
        # Filter out single chars and stopword-like tokens
        keywords = [k for k in keywords if len(k) > 2][:n]
        topics[label] = keywords
    return topics


def _generate_insight(results, title=''):
    """Rule-based 2–3 sentence insight summary."""
    total = len(results)
    if total == 0:
        return "No comments were analyzed."
    c = {'1': 0, '0': 0, '-1': 0}
    toxic = 0
    spam = 0
    for r in results:
        c[r.get('sentiment', '0')] = c.get(r.get('sentiment', '0'), 0) + 1
        if r.get('is_toxic'):
            toxic += 1
        if r.get('is_spam'):
            spam += 1
    pos_pct = round(c['1'] / total * 100)
    neu_pct = round(c['0'] / total * 100)
    neg_pct = round(c['-1'] / total * 100)

    dominant = max(c, key=c.get)
    dominant_label = {'1': 'positive', '0': 'neutral', '-1': 'negative'}[dominant]

    parts = []
    parts.append(
        f"Out of {total} analyzed comments, {pos_pct}% are positive, "
        f"{neu_pct}% neutral, and {neg_pct}% negative."
    )
    if dominant == '1':
        parts.append("The audience reception is largely favorable.")
    elif dominant == '-1':
        parts.append("The audience is predominantly critical — consider reviewing the feedback carefully.")
    else:
        parts.append("Audience sentiment is mixed, with no strong lean in either direction.")

    flags = []
    if toxic > 0:
        flags.append(f"{toxic} toxic comment{'s' if toxic > 1 else ''} ({round(toxic/total*100)}%)")
    if spam > 0:
        flags.append(f"{spam} likely spam/low-quality comment{'s' if spam > 1 else ''}")
    if flags:
        parts.append("Watch out for: " + " and ".join(flags) + ".")

    return " ".join(parts)


def _predict(raw_items):
    """Run sentiment prediction; adds confidence, lang, toxicity, spam per item."""
    import numpy as np
    texts = [c['text'] for c in raw_items]
    preprocessed = [preprocess_comment(t) for t in texts]
    transformed = vectorizer.transform(preprocessed)
    sentiments = [str(p) for p in model.predict(transformed).tolist()]
    probas = model.predict_proba(transformed)   # shape (n, 3)
    results = []
    for c, s, prob in zip(raw_items, sentiments, probas):
        tox = _toxicity_score(c['text'])
        is_spam, spam_reason = _spam_score(c['text'])
        confidence = float(round(float(np.max(prob)), 3))
        results.append({
            "comment":        c['text'],
            "sentiment":      s,
            "confidence":     confidence,
            "uncertain":      confidence < 0.55,
            "votes":          c['votes'],
            "timestamp":      c['timestamp'],
            "lang":           _detect_lang(c['text']),
            "toxicity_score": tox,
            "is_toxic":       tox >= 5,
            "is_spam":        is_spam,
            "spam_reason":    spam_reason,
        })
    return results


def _oembed(video_id):
    """Return (title, channel) for a video ID via YouTube oEmbed."""
    try:
        data = http_requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=5
        ).json()
        return data.get('title', 'Unknown Title'), data.get('author_name', 'Unknown Channel')
    except Exception:
        return 'Unknown Title', 'Unknown Channel'


def _channel_video_urls(channel_input, max_videos):
    """Return a list of video URLs from a channel URL or ID."""
    channel_id = None

    # Direct /channel/UC... URL
    m = re.search(r'/channel/(UC[A-Za-z0-9_-]+)', channel_input)
    if m:
        channel_id = m.group(1)

    if not channel_id:
        # Normalise: strip trailing path segments like /videos, /shorts, /streams
        base_url = re.sub(r'/(videos|shorts|streams|playlists|community|about|featured)\s*$', '', channel_input.strip().rstrip('/'))
        # Also try without query params
        base_url = base_url.split('?')[0]

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        for try_url in [base_url, channel_input.strip()]:
            try:
                page = http_requests.get(try_url, headers=headers, timeout=12).text
                for pat in [
                    r'"channelId"\s*:\s*"(UC[A-Za-z0-9_-]+)"',
                    r'"externalId"\s*:\s*"(UC[A-Za-z0-9_-]+)"',
                    r'"browseId"\s*:\s*"(UC[A-Za-z0-9_-]+)"',
                    r'"ucid"\s*:\s*"(UC[A-Za-z0-9_-]+)"',
                    r'href="/channel/(UC[A-Za-z0-9_-]+)"',
                    r'channel/(UC[A-Za-z0-9_-]+)',
                ]:
                    m2 = re.search(pat, page)
                    if m2:
                        channel_id = m2.group(1)
                        break
                if channel_id:
                    break
            except Exception:
                pass

    if not channel_id:
        return []
    try:
        rss = http_requests.get(
            f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}", timeout=10
        )
        root = ET.fromstring(rss.text)
        ns = {'yt': 'http://www.youtube.com/xml/schemas/2015',
              'atom': 'http://www.w3.org/2005/Atom'}
        urls = []
        for entry in root.findall('atom:entry', ns)[:max_videos]:
            vid = entry.find('yt:videoId', ns)
            if vid is not None:
                urls.append(f"https://www.youtube.com/watch?v={vid.text}")
        return urls
    except Exception:
        return []


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    data = request.json
    url = data.get('url', '').strip()
    max_comments = int(data.get('max_comments', 100))
    fetch_all = max_comments <= 0          # 0 = "all comments"
    sort_mode = data.get('sort_by', 'top')

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    m = re.search(r'(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})', url)
    if not m:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    video_id = m.group(1)

    title, channel = _oembed(video_id)

    sort_flag = SORT_BY_RECENT if sort_mode == 'recent' else SORT_BY_POPULAR
    if fetch_all:
        fetch_limit = None          # no cap — exhaust the generator
    elif sort_mode == 'random':
        fetch_limit = max_comments * 3
    else:
        fetch_limit = max_comments

    try:
        raw = _fetch_comments(url, sort_flag, fetch_limit)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch comments: {str(e)}"}), 500

    if not raw:
        return jsonify({"error": "No comments found for this video"}), 404

    if sort_mode == 'random' and not fetch_all and len(raw) > max_comments:
        raw = random.sample(raw, max_comments)

    try:
        results = _predict(raw)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    topics  = _extract_topics(results)
    insight = _generate_insight(results, title)

    return jsonify({"video_id": video_id, "title": title, "channel": channel,
                    "total": len(results), "results": results,
                    "topics": topics, "insight": insight})


@app.route('/get_topics', methods=['POST'])
def get_topics():
    """Extract top TF-IDF keywords per sentiment group from a results array."""
    data = request.json
    results = data.get('results', [])
    if not results:
        return jsonify({"error": "No results provided"}), 400
    return jsonify(_extract_topics(results))


@app.route('/generate_insight', methods=['POST'])
def generate_insight():
    """Return a rule-based insight summary for a results array."""
    data = request.json
    results = data.get('results', [])
    title   = data.get('title', '')
    if not results:
        return jsonify({"error": "No results provided"}), 400
    return jsonify({"insight": _generate_insight(results, title)})


@app.route('/analyze_channel', methods=['POST'])
def analyze_channel():
    data = request.json
    channel_input = data.get('channel', '').strip()
    max_per_video = min(int(data.get('max_per_video', 50)), 200)
    max_videos    = min(int(data.get('max_videos', 5)), 10)

    if not channel_input:
        return jsonify({"error": "No channel URL provided"}), 400

    video_urls = _channel_video_urls(channel_input, max_videos)
    if not video_urls:
        return jsonify({"error": "Could not find videos for this channel. Try a direct /channel/UC... URL."}), 404

    video_analyses = []
    all_results = []

    for vurl in video_urls:
        vm = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', vurl)
        if not vm:
            continue
        vid = vm.group(1)
        vtitle, vchannel = _oembed(vid)
        try:
            raw = _fetch_comments(vurl, SORT_BY_POPULAR, max_per_video)
        except Exception:
            continue
        if not raw:
            continue
        try:
            results = _predict(raw)
        except Exception:
            continue

        counts = {'1': 0, '0': 0, '-1': 0}
        for r in results:
            counts[r['sentiment']] = counts.get(r['sentiment'], 0) + 1
        total = len(results)

        # top 5 comments per sentiment for this video
        def top_by_sent(sent, n=5):
            return sorted(
                [r for r in results if r['sentiment'] == sent],
                key=lambda x: x.get('votes', 0), reverse=True
            )[:n]

        video_analyses.append({
            'video_id': vid, 'title': vtitle, 'channel': vchannel, 'total': total,
            'positive': counts['1'], 'neutral': counts['0'], 'negative': counts['-1'],
            'score': round((counts['1'] - counts['-1']) / total * 100) if total else 0,
            'top_positive': top_by_sent('1'),
            'top_neutral':  top_by_sent('0'),
            'top_negative': top_by_sent('-1'),
        })
        all_results.extend(results)

    if not video_analyses:
        return jsonify({"error": "No videos could be analyzed"}), 500

    total_all = sum(v['total'] for v in video_analyses)
    pos_all   = sum(v['positive'] for v in video_analyses)
    neu_all   = sum(v['neutral']  for v in video_analyses)
    neg_all   = sum(v['negative'] for v in video_analyses)
    health    = round((pos_all - neg_all) / total_all * 100) if total_all else 0

    channel_name = video_analyses[0].get('channel', 'Channel')
    insight  = _generate_insight(all_results, channel_name)
    topics   = _extract_topics(all_results)

    # channel-wide top comments per sentiment
    def ch_top(sent, n=5):
        return sorted(
            [r for r in all_results if r['sentiment'] == sent],
            key=lambda x: x.get('votes', 0), reverse=True
        )[:n]

    return jsonify({
        'channel': channel_name,
        'videos_analyzed': len(video_analyses),
        'health_score': health,
        'total_comments': total_all,
        'positive': pos_all, 'neutral': neu_all, 'negative': neg_all,
        'videos': video_analyses,
        'results': all_results,
        'insight': insight,
        'topics': topics,
        'top_positive': ch_top('1'),
        'top_neutral':  ch_top('0'),
        'top_negative': ch_top('-1'),
    })


@app.route('/save_report', methods=['POST'])
def save_report():
    os.makedirs('reports', exist_ok=True)
    slug = uuid.uuid4().hex[:8]
    with open(f'reports/{slug}.json', 'w') as f:
        json.dump(request.json, f)
    return jsonify({'slug': slug})


@app.route('/get_report/<slug>', methods=['GET'])
def get_report(slug):
    if not re.match(r'^[a-f0-9]{8}$', slug):
        return jsonify({'error': 'Invalid report ID'}), 400
    path = f'reports/{slug}.json'
    if not os.path.exists(path):
        return jsonify({'error': 'Report not found'}), 404
    with open(path) as f:
        return jsonify(json.load(f))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
