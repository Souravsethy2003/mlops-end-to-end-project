# SentimentScope — YouTube Comment Sentiment Analysis

A production-grade MLOps project that analyzes YouTube comment sentiment in real time. Built as a final year project, it features a full DVC/MLflow pipeline, a Flask REST API, and a dark-themed single-page frontend.

**Live demo:** `http://3.110.41.116/app`

---

## Features

- **4 analysis modes** — Single video, manual text, full channel, side-by-side comparison
- **Sentiment classification** — Positive / Neutral / Negative (3-class)
- **Per-comment metadata** — confidence score, toxicity score, spam detection, language detection, vote count
- **Visualizations** — donut chart, word cloud, sentiment trend graph, emoji map, age analysis, influential comments
- **Topic extraction** — top TF-IDF keywords per sentiment group
- **AI insights** — rule-based 2–3 sentence summary of comment section
- **Export** — CSV download, shareable report links (slug-based)
- **MLflow tracking** — all experiments, metrics, and model versions logged

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Nginx (port 80)             │
│  /app  → frontend/index.html (static)        │
│  /     → Flask API (port 5000)               │
│  /mlflow/ → MLflow UI (port 5001)            │
└─────────────────────────────────────────────┘
         │                        │
   ┌─────▼──────┐          ┌──────▼──────┐
   │  Flask API  │          │   MLflow    │
   │  port 5000  │          │  port 5001  │
   │  app.py     │          │  mlflow.db  │
   └─────┬───────┘          └─────────────┘
         │
   ┌─────▼──────────────────────┐
   │  LightGBM + TF-IDF Model   │
   │  yt_chrome_plugin_model v4  │
   │  Macro F1: 0.848            │
   └────────────────────────────┘
```

---

## ML Pipeline

Managed with **DVC** (5 stages) and tracked with **MLflow**.

```
data_ingestion → data_preprocessing → model_building → model_evaluation → model_registration
```

| Stage | Script | Output |
|-------|--------|--------|
| Data Ingestion | `src/data/data_ingestion.py` | `data/raw/train.csv`, `data/raw/test.csv` |
| Preprocessing | `src/data/data_preprocessing.py` | `data/interim/train_processed.csv` |
| Model Building | `src/model/model_building.py` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| Evaluation | `src/model/model_evaluation.py` | `experiment_info.json`, confusion matrix |
| Registration | `src/model/register_model.py` | MLflow Model Registry |

**Run the full pipeline:**
```bash
dvc repro
```

---

## Model

| Property | Value |
|----------|-------|
| Algorithm | LightGBM (multiclass) |
| Features | TF-IDF, max 10K features, 1–3 grams |
| Classes | -1 (Negative), 0 (Neutral), 1 (Positive) |
| Training data | ~37K Reddit comments (clean, pre-labeled) |
| Test Accuracy | **85.95%** |
| Macro F1 | **0.848** |
| MLflow model name | `yt_chrome_plugin_model` |
| Active version | v4 |

**Per-class performance (v4):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.796 | 0.758 | 0.776 |
| Neutral | 0.847 | 0.962 | 0.901 |
| Positive | 0.907 | 0.830 | 0.867 |

---

## Training Data

The `data_ingestion.py` script supports multiple data sources, configurable via `params.yaml`:

| Source | Size | Domain | Used for |
|--------|------|--------|---------|
| Reddit (Himanshu-1703) | 37K | Social media | LightGBM training |
| TweetEval sentiment | 60K | Twitter | BERT training |
| SST-2 (Stanford) | 68K | Movie reviews | BERT training |
| GoEmotions (Google) | 43K | Reddit | BERT training |
| YouTube (self-scraped) | 7.6K | YouTube comments | BERT training |

To scrape fresh YouTube comments for training:
```bash
python scripts/scrape_youtube_training_data.py
```

---

## API Endpoints

Base URL: `http://3.110.41.116`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict sentiment for a list of comments |
| POST | `/predict_with_timestamps` | Predict with timestamp data |
| POST | `/analyze_video` | Full YouTube video analysis (fetches + predicts) |
| POST | `/analyze_channel` | Multi-video channel analysis |
| POST | `/generate_wordcloud` | Generate word cloud image (PNG) |
| POST | `/generate_trend_graph` | Generate monthly trend graph (PNG) |
| POST | `/generate_chart` | Generate sentiment pie chart (PNG) |
| POST | `/get_topics` | Extract TF-IDF keywords per sentiment group |
| POST | `/generate_insight` | Generate rule-based insight summary |
| POST | `/save_report` | Save analysis results, returns shareable slug |
| GET | `/get_report/<slug>` | Retrieve saved report by slug |

**Example — analyze a video:**
```bash
curl -X POST http://3.110.41.116/analyze_video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "max_comments": 100, "sort_by": "top"}'
```

**Example — predict comments:**
```bash
curl -X POST http://3.110.41.116/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This is amazing!", "Terrible video", "It was okay"]}'
```

---

## Project Structure

```
Mlflow/
├── flask_app/
│   ├── app.py                  # Flask API (12 endpoints)
│   ├── requirements.txt        # Flask-specific dependencies
│   └── reports/                # Saved analysis reports (JSON)
├── frontend/
│   └── index.html              # Single-page frontend (dark theme)
├── src/
│   ├── data/
│   │   ├── data_ingestion.py   # Multi-source data loading
│   │   └── data_preprocessing.py
│   └── model/
│       ├── model_building.py   # TF-IDF + LightGBM training
│       ├── model_evaluation.py # MLflow logging + confusion matrix
│       └── register_model.py   # MLflow Model Registry
├── scripts/
│   ├── scrape_youtube_training_data.py  # YouTube comment scraper
│   ├── test_flask_api.py
│   ├── test_model_performance.py
│   ├── test_model_signature.py
│   ├── test_load_model.py
│   ├── promote_model.py        # Staging → Production promotion
│   └── mlflow_test.py
├── plan/
│   └── PLAN.md                 # 7-phase enhancement roadmap
├── data/
│   ├── raw/                    # Train/test CSV (DVC-tracked)
│   ├── interim/                # Preprocessed CSV (DVC-tracked)
│   └── external/               # YouTube scraped comments
├── mlartifacts/                # MLflow artifact storage
├── params.yaml                 # Pipeline hyperparameters
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC reproducibility lock
├── start.sh                    # Single command to start all services
├── Dockerfile                  # Docker image for Flask API
├── appspec.yml                 # AWS CodeDeploy config
├── lgbm_model.pkl              # Trained LightGBM model
├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
└── mlflow.db                   # MLflow SQLite tracking database
```

---

## Quick Start

**1. Clone and set up environment:**
```bash
git clone https://github.com/geekylakshya/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Download NLTK data:**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

**3. Start all services (single command):**
```bash
bash start.sh
```

**4. Open the app:**
- Frontend: `http://3.110.41.116/app`
- MLflow UI: `http://3.110.41.116/mlflow/`

---

## Reproducing the Pipeline

```bash
# Run full DVC pipeline (ingestion → preprocessing → training → evaluation → registration)
dvc repro

# Or run individual stages
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

---

## Testing

```bash
# Test Flask API endpoints
pytest scripts/test_flask_api.py

# Test model performance thresholds
pytest scripts/test_model_performance.py

# Test model loading from MLflow
python scripts/test_load_model.py

# Test model signature
python scripts/test_model_signature.py
```

---

## Deployment

The project deploys via **AWS CodeDeploy** with Docker:

```bash
# Build Docker image
docker build -t yt-sentiment .

# Run container
docker run -p 5000:5000 yt-sentiment
```

The `appspec.yml` automates deployment to EC2 by pulling from ECR and restarting the container.

---

## Enhancement Roadmap

See [`plan/PLAN.md`](plan/PLAN.md) for the full 7-phase roadmap including:

- **Phase 1** — DistilBERT fine-tuning + model comparison notebook
- **Phase 2** — Emotion detection + aspect-based sentiment
- **Phase 3** — Redis caching + Celery async tasks
- **Phase 4** — PDF report export + new visualizations
- **Phase 5** — A/B testing framework + prediction monitoring
- **Phase 6** — Full evaluation notebook + academic presentation polish

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | LightGBM + TF-IDF (scikit-learn) |
| Experiment Tracking | MLflow 2.17 |
| Pipeline Orchestration | DVC 3.53 |
| API | Flask 3.0 + Flask-CORS |
| Frontend | Vanilla JS + Chart.js (single HTML) |
| Web Server | Nginx |
| Deployment | Docker + AWS CodeDeploy + ECR |
| NLP | NLTK (stopwords, lemmatization) + langdetect |
| Visualization | Matplotlib, WordCloud, Chart.js |
