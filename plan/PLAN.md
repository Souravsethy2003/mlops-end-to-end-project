# SentimentScope Enhancement Plan — Final Year Project

## Context

SentimentScope is a YouTube comment sentiment analysis platform with a LightGBM + TF-IDF pipeline, Flask API, and a dark-themed frontend. For a final year project, the biggest gap is **model sophistication** — a single gradient-boosted tree on TF-IDF features is baseline-level. Adding a BERT-based model with proper comparison, plus richer analysis features, transforms this into a research-grade project.

**Current stack**: LightGBM (3-class) / TF-IDF (10K features, 1-3 grams) / MLflow / DVC / Flask / Nginx / Single-page frontend

**Current data**: ~30K Reddit comments (42% positive, 34% neutral, 22% negative) — too small and single-domain for robust sentiment analysis.

---

## Phase 0: Dataset Expansion & Improvement (Priority: CRITICAL — do this FIRST)

> More data = better models. Currently 30K Reddit-only comments is too small and creates domain mismatch (trained on Reddit, used on YouTube). This phase 3-5x the dataset size and adds YouTube-domain data.

### 0.1 Add Public Sentiment Datasets

**Modify**: `src/data/data_ingestion.py`

Add these publicly available datasets (all free, no API keys needed):

| Dataset | Source | Size | Format | Why |
|---------|--------|------|--------|-----|
| **Sentiment140** | [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) | 1.6M tweets | text, label (0=neg, 4=pos) | Massive social media text, similar to YT comments |
| **Amazon Reviews** | [HuggingFace](https://huggingface.co/datasets/mteb/amazon_polarity) | 3.6M reviews | content, label (1=neg, 2=pos) | Product sentiment, diverse vocabulary |
| **SST-2 (Stanford)** | [HuggingFace](https://huggingface.co/datasets/stanfordnlp/sst2) | 67K sentences | sentence, label (0=neg, 1=pos) | Academic gold standard |
| **GoEmotions** | [HuggingFace](https://huggingface.co/datasets/google-research-datasets/go_emotions) | 58K Reddit | text, 28 emotion labels | Reddit + emotion labels (reusable in Phase 2) |
| **YouTube Comments** | Self-scraped | 10-50K | text, model-predicted + manual | Closes the domain gap |

**Recommended strategy** — don't just dump everything in:

1. **Core training set (target: 150K-200K balanced):**
   - Keep existing 30K Reddit comments
   - Add 40K from Sentiment140 (sample 20K negative, 20K positive, map to -1/1)
   - Add 30K from Amazon Reviews (sample balanced, map to -1/1)
   - Add SST-2 full (67K, map 0->-1, 1->1)
   - Add GoEmotions (map emotion labels to sentiment: joy/love/admiration->1, anger/disgust/sadness->-1, neutral->0)
   - **For neutral class**: Sample from Sentiment140/Amazon where model confidence is low (ambiguous texts tend to be neutral-ish)

2. **YouTube domain adaptation set (target: 20-50K):**
   - Use the existing `/analyze_video` endpoint to scrape and predict comments from 50-100 popular videos across genres
   - Manual spot-check and correction of a subset (1-2K comments) for a clean YouTube test set
   - This creates a YouTube-specific evaluation set (critical for demonstrating the app works on its actual domain)

### 0.2 Data Pipeline Update

**Modify**: `src/data/data_ingestion.py`

```python
# New function structure:
def load_reddit_data():        # existing source
def load_sentiment140():       # download from Kaggle/mirror
def load_amazon_reviews():     # via HuggingFace datasets lib
def load_sst2():               # via HuggingFace datasets lib  
def load_goemotions():         # via HuggingFace datasets lib
def load_youtube_scraped():    # from data/external/youtube_comments.csv
def merge_and_balance():       # combine all, balance classes, deduplicate
```

Key implementation details:
- Map all labels to the same schema: `-1` (negative), `0` (neutral), `1` (positive)
- Add a `source` column to track which dataset each row came from (useful for analysis)
- Balance classes via undersampling the majority class or oversampling the minority class (negative is currently only 22%)
- Deduplicate by text hash
- Keep the same train/test split ratio (80/20) with stratification

**Modify**: `params.yaml` — add:
```yaml
data_ingestion:
  test_size: 0.20
  use_sentiment140: true
  sentiment140_sample: 40000
  use_amazon: true
  amazon_sample: 30000
  use_sst2: true
  use_goemotions: true
  use_youtube_scraped: true
  balance_strategy: "undersample"  # or "oversample" or "none"
```

**Modify**: `dvc.yaml` — update data_ingestion stage dependencies to include new params

### 0.3 YouTube Comment Scraping Script

**New file**: `scripts/scrape_youtube_training_data.py`

- Scrape comments from 50-100 popular YouTube videos across diverse genres (music, tech, politics, gaming, vlogs, education)
- Use the existing `youtube_comment_downloader` already in the project
- Save to `data/external/youtube_comments.csv` with columns: `clean_comment`, `category` (predicted by current model), `video_id`, `source_genre`
- Target: 20-50K comments
- Run existing model to auto-label, then manually review a subset

**New file**: `data/external/youtube_comments.csv` — output of scraping

### 0.4 Data Quality & Analysis Notebook

**New file**: `notebooks/data_analysis.ipynb`

This is important for academic presentation:
1. Dataset composition breakdown (pie chart of sources)
2. Class distribution before/after balancing
3. Text length distributions across sources
4. Vocabulary overlap between Reddit, Twitter, YouTube, Amazon
5. Domain shift analysis: word frequency differences across sources
6. t-SNE or UMAP visualization of TF-IDF embeddings colored by source
7. Before/after model performance showing data augmentation impact

### 0.5 Expected Impact

| Metric | Before (30K Reddit) | After (~150K multi-source) |
|--------|---------------------|---------------------------|
| Dataset size | 30K | 150-200K |
| Sources | 1 (Reddit) | 5+ (Reddit, Twitter, Amazon, SST, YouTube) |
| Class balance | 42/34/22 | ~33/33/33 |
| Domain match | Low (Reddit != YouTube) | High (includes YouTube data) |
| Negative class | 6.5K samples | ~50K samples |

**Time estimate**: 1-1.5 weeks

---

## Phase 1: BERT Model Integration & Comparative Study (Priority: CRITICAL)

> This is the single most impactful addition for academic evaluation.

### 1.1 Fine-tune DistilBERT

**New files:**
- `src/model/bert_model_building.py` — Fine-tune `distilbert-base-uncased` on the same dataset
- `src/model/bert_model_evaluation.py` — Evaluate and log metrics to MLflow
- `src/model/bert_register_model.py` — Register as `yt_bert_sentiment_model`

**Key decisions:**
- Use **raw text** (not TF-IDF preprocessed) for BERT — BERT benefits from full context including stopwords
- Load from `data/raw/train.csv` directly, not `data/interim/`
- DistilBERT, not full BERT — 2x faster, 40% smaller, ~97% of BERT's performance
- CPU inference is fine (~50-100 comments/sec with batching)
- Map labels: -1->0, 0->1, 1->2 (HuggingFace expects 0-indexed)

**Modify `params.yaml`** — add:
```yaml
bert_model:
  learning_rate: 2e-5
  epochs: 3
  batch_size: 16
  max_length: 128
  warmup_steps: 500
  weight_decay: 0.01
```

**Modify `dvc.yaml`** — add 3 stages: `bert_model_building`, `bert_model_evaluation`, `bert_model_registration`

**Dependencies to add:**
```
transformers>=4.36.0
torch>=2.1.0
datasets>=2.16.0
accelerate>=0.25.0
```

**Output**: Model saved to `models/bert_sentiment/` (~250MB)

### 1.2 Model Comparison Notebook

**New file**: `notebooks/model_comparison.ipynb`

Contents:
1. Quantitative comparison table (accuracy, macro-F1, weighted-F1, per-class P/R/F1)
2. Side-by-side confusion matrices
3. Error analysis — comments where models disagree, categorized by pattern (sarcasm, negation, slang)
4. Confidence calibration curves (`sklearn.calibration.calibration_curve`)
5. Inference speed benchmark (LightGBM will be 10-100x faster)
6. Ablation: BERT with different max_length (64/128/256), epochs (1-5)
7. Statistical significance: McNemar's test on paired predictions
8. Cross-domain analysis: trained on Reddit, tested on YouTube — discuss domain shift

### 1.3 Dual-Model API

**Modify**: `flask_app/app.py`

- Add `load_bert_model()` — loads DistilBERT from `models/bert_sentiment/`
- Add `_predict_bert(raw_items)` — tokenize + inference, same output format as `_predict()`
- Add `_predict_ensemble(raw_items)` — weighted average of probabilities (0.6 BERT + 0.4 LightGBM)
- Add `model` param to `/analyze_video`, `/predict`: `model=lgbm|bert|ensemble`
- Include `model_used` field in responses
- New endpoint: `POST /model_comparison` — runs both models, returns side-by-side with agreement flags

### 1.4 Frontend Model Selector

**Modify**: `frontend/index.html`

- Add dropdown in input area: "LightGBM (Fast)" / "DistilBERT (Accurate)" / "Ensemble (Best)"
- Pass selected model to API calls
- Show `model_used` in results header
- When ensemble: show both confidence values per comment

---

## Phase 2: Emotion Detection & Aspect-Based Sentiment

### 2.1 Emotion Detection (Zero-Shot)

**New file**: `flask_app/emotion_detector.py`

- Use HuggingFace zero-shot pipeline with `typeform/distilbart-mnli-12-3` (lighter than bart-large)
- Emotions: joy, anger, sadness, surprise, fear, disgust, love, neutral
- Function `detect_emotions_batch(texts)` returns top 3 emotions per text with scores
- Singleton model loading (same pattern as existing model/vectorizer)

**Modify**: `flask_app/app.py`
- Add `emotions` field to prediction results: `{"joy": 0.72, "surprise": 0.15}`
- New endpoint: `POST /analyze_emotions`

**Modify**: `frontend/index.html`
- New viz: **Emotion Radar Chart** (Chart.js radar type) — aggregate emotion distribution
- New viz: **Emotion Heatmap** — emotion intensity by sentiment bucket
- Per-comment: small colored emotion chips next to sentiment badges

### 2.2 Aspect-Based Sentiment

**New file**: `flask_app/aspect_sentiment.py`

- Extract noun phrases using NLTK POS tagging
- Group comments by detected aspects (e.g., "audio", "editing", "content")
- Run sentiment on each aspect group
- Returns: `{aspect: {count, avg_sentiment, top_comments}}`

**Modify**: `flask_app/app.py`
- Integrate into `/analyze_video` response
- New endpoint: `POST /aspects`

**Modify**: `frontend/index.html`
- New section: **Aspect Sentiment Panel** — horizontal bar chart per aspect with pos/neg split
- Click aspect to filter comment list

---

## Phase 3: Async Processing & Caching

### 3.1 Redis Caching

**New file**: `flask_app/cache.py`

- Redis connection with graceful fallback if unavailable
- Cache key: `video:{video_id}:{max_comments}:{sort_by}:{model}`
- TTL: 1 hour (video), 24 hours (channel)
- Wrap `analyze_video` and `analyze_channel` with caching

**Dependency**: `redis>=5.0.0`

### 3.2 Celery Async Tasks

**New file**: `flask_app/tasks.py`

- Celery app with Redis broker
- Task `analyze_video_async()` for large jobs
- Task `analyze_channel_async()` for channel analysis

**Modify**: `flask_app/app.py`
- `POST /analyze_video_async` — returns `{"task_id": "..."}`
- `GET /task_status/<task_id>` — returns state + result
- Synchronous endpoints remain for backward compat

**Modify**: `frontend/index.html`
- Use async endpoint for "Fetch All" (1000+ comments)
- Poll `/task_status/` with progress indicator

### 3.3 Rate Limiting

- Add `flask-limiter` (30 req/min per IP)
- Pydantic request validation at API boundary

**Dependency**: `flask-limiter>=3.5.0`

---

## Phase 4: Enhanced Visualizations & PDF Export

### 4.1 New Frontend Charts

**Modify**: `frontend/index.html`

- **Confidence Histogram** — Chart.js bar chart (buckets: 50-60%, 60-70%, etc.)
- **Language Distribution Pie** — using existing `lang` field
- **Comment Length vs Sentiment Scatter** — Chart.js scatter plot
- **Sentiment Shift Alerts** — highlight months with >20% sentiment change

### 4.2 PDF Report Generation

**New file**: `flask_app/report_generator.py`

- Use `reportlab` to build professional PDF
- Sections: Executive Summary, Sentiment Distribution (donut chart embedded), Trend Analysis, Emotion Breakdown, Aspect Analysis, Top Comments, Toxicity Summary, Model Info
- SentimentScope branding

**Modify**: `flask_app/app.py`
- `POST /generate_pdf_report` — returns downloadable PDF

**Modify**: `frontend/index.html`
- "Download PDF Report" button next to existing CSV export

**Dependency**: `reportlab>=4.0.0`

---

## Phase 5: MLOps & Model A/B Testing

### 5.1 A/B Testing Framework

**New file**: `flask_app/ab_testing.py`

- Route requests to LightGBM or BERT based on configurable split (e.g., 70/30)
- Log model assignments to SQLite (`flask_app/ab_logs.db`)
- `GET /ab_status` — current config + aggregate metrics
- `POST /ab_config` — update split ratio
- `model=auto` in `/analyze_video` uses the A/B router

### 5.2 Prediction Monitoring

**New file**: `flask_app/monitoring.py`

- Track prediction distribution over time (detect drift)
- Track latency per endpoint (p50, p95, p99)
- Track confidence distributions
- `GET /monitoring/dashboard` — drift metrics, latency, distribution over 24h/7d/30d

### 5.3 BERT Test Scripts

**New file**: `scripts/test_bert_model_performance.py`
- Same structure as `scripts/test_model_performance.py` for BERT model
- Higher thresholds (BERT should outperform LightGBM)

**Modify**: `Dockerfile`
- Add `COPY models/bert_sentiment/ /app/models/bert_sentiment/`
- Consider multi-stage build for image size

---

## Phase 6: Polish & Academic Presentation

### 6.1 Comprehensive Evaluation Notebook

**New file**: `notebooks/full_evaluation.ipynb`

1. Dataset statistics (class distribution, comment length, vocabulary)
2. Preprocessing pipeline visualization (before/after)
3. LightGBM feature importance (top 50 TF-IDF features per class)
4. BERT attention visualization (using `bertviz`)
5. Full comparison table from MLflow
6. Ensemble analysis: when does ensemble beat individuals?
7. Error analysis deep dive with categorized failures
8. Ablation studies
9. Cross-domain (Reddit->YouTube) analysis

**Dependency**: `bertviz>=1.4.0`

### 6.2 Frontend Polish

**Modify**: `frontend/index.html`
- "Model Info" panel showing active model, dataset size, metrics, last trained
- Loading skeletons (CSS pulse) instead of basic spinners
- Dark/light theme toggle

### 6.3 README Update

**Modify**: `README.md`
- Architecture diagram
- Full API docs
- Setup instructions for all services
- Model performance summary

---

## Timeline & Priority

| Phase | Time | Academic Impact | Priority |
|-------|------|----------------|----------|
| **Phase 0: Data Expansion** | 1-1.5 weeks | Very High | Must do FIRST |
| Phase 1: BERT + Comparison | 2-3 weeks | Very High | Must do |
| Phase 2: Emotion + Aspect | 1.5-2 weeks | High | Should do |
| Phase 3: Async + Caching | 1-1.5 weeks | Medium | Nice to have |
| Phase 4: Viz + PDF Export | 1.5-2 weeks | High | Should do |
| Phase 5: MLOps + A/B | 1.5 weeks | High | Should do |
| Phase 6: Polish + Docs | 1 week | Very High | Must do |

**Total: ~10-13 weeks**

**If time is limited, do**: Phase 0 -> Phase 1 -> Phase 6 -> Phase 4 -> Phase 2 -> Phase 5 -> Phase 3

Phase 0 must come first because both LightGBM and BERT benefit from more data, and it enables the "before/after data augmentation" comparison in the evaluation notebook.

---

## New Files Summary

```
scripts/scrape_youtube_training_data.py # YouTube comment scraper for training data
data/external/youtube_comments.csv     # Scraped YouTube comments
notebooks/data_analysis.ipynb          # Dataset composition & domain analysis
src/model/bert_model_building.py       # BERT fine-tuning
src/model/bert_model_evaluation.py     # BERT evaluation + MLflow
src/model/bert_register_model.py       # BERT registry
flask_app/emotion_detector.py          # Zero-shot emotion detection
flask_app/aspect_sentiment.py          # Aspect-based analysis
flask_app/cache.py                     # Redis caching layer
flask_app/tasks.py                     # Celery async tasks
flask_app/ab_testing.py                # A/B testing router
flask_app/monitoring.py                # Prediction monitoring
flask_app/report_generator.py          # PDF report generation
scripts/test_bert_model_performance.py # BERT test script
notebooks/model_comparison.ipynb       # Model comparison analysis
notebooks/full_evaluation.ipynb        # Complete evaluation
models/bert_sentiment/                 # Fine-tuned BERT weights
```

## Files to Modify

```
flask_app/app.py                       # Dual-model prediction, new endpoints
frontend/index.html                    # Model selector, new visualizations
params.yaml                            # BERT hyperparameters
dvc.yaml                               # BERT pipeline stages
requirements.txt                       # New dependencies
flask_app/requirements.txt             # New dependencies
Dockerfile                             # Include BERT model
README.md                              # Documentation
```

## Verification

After each phase:
1. Run `dvc repro` to verify pipeline (Phase 1)
2. Run `python scripts/test_bert_model_performance.py` (Phase 1)
3. Test all API endpoints: `curl -X POST http://localhost:5000/analyze_video -H "Content-Type: application/json" -d '{"url":"https://youtube.com/watch?v=XXXXX","max_comments":10,"model":"bert"}'`
4. Open `http://15.207.222.63/app` and verify all frontend features
5. Run comparison notebook and verify all cells execute
6. Test PDF export downloads correctly
