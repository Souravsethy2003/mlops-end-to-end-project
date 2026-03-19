# BERT Sentiment Model — Training Plan for Google Colab

## 1. Project Context

This is a **3-class sentiment analysis** system for social media text (YouTube comments, Reddit posts, tweets). The production system already has a LightGBM + TF-IDF model deployed in a Flask API. This BERT model will be added as a **second inference option** alongside LightGBM, selectable by the user via the API.

**Labels:**
| Value | Meaning |
|-------|---------|
| `-1`  | Negative |
| `0`   | Neutral  |
| `1`   | Positive |

---

## 2. Dataset

### Files (upload both to Colab or mount via Google Drive)

| File | Rows | Description |
|------|------|-------------|
| `bert_train.csv` | 77,120 | Training set (80% stratified split) |
| `bert_test.csv`  | 19,281 | Test set (20% stratified split) |

### Columns
```
text    →  cleaned social media text (tweets + Reddit comments)
label   →  -1, 0, or 1
source  →  "TweetEval" or "reddit"
```

### Class Distribution (same in both train and test — stratified split)
| Label | Class | Count (train) | % |
|-------|-------|---------------|---|
| -1 | Negative | 15,687 | 20.3% |
|  0 | Neutral  | 32,017 | 41.5% |
|  1 | Positive | 29,416 | 38.1% |

### Data Sources
- **TweetEval** (cardiffnlp/tweet_eval, sentiment config, HuggingFace) — Twitter tweets, professionally annotated
- **Reddit** (Himanshu-1703/reddit-sentiment-analysis, GitHub) — Reddit comments

### Important Notes on the Data
- Text has already been cleaned: @mentions anonymised to `@user`, URLs stripped, RT markers removed
- Both sources use identical label values (-1, 0, 1)
- Duplicates have been removed
- The neutral class dominates at 41.5% — **must be mitigated during training** (see Section 5)

---

## 3. Model Choice

**Model: `distilbert-base-uncased`**

- 40% smaller than BERT-base, 60% faster, retains 97% of BERT performance
- 66M parameters — fits comfortably in Colab T4 GPU (15GB VRAM)
- Well-suited for short social media text
- HuggingFace model ID: `distilbert-base-uncased`

**Architecture modification:** Add a classification head on top of the `[CLS]` token representation:
- `DistilBertForSequenceClassification` with `num_labels=3`
- Output layer: Linear(768 → 3) with softmax

---

## 4. Environment Setup (Colab)

```python
# Install required packages
!pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn

# Verify GPU
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
# Must show GPU — if not, Runtime > Change runtime type > T4 GPU

# Mount Google Drive (if uploading files via Drive)
from google.colab import drive
drive.mount('/content/drive')

# Or upload directly
from google.colab import files
uploaded = files.upload()  # upload bert_train.csv and bert_test.csv
```

---

## 5. Class Imbalance Strategy — Handling Neutral Dominance

### Problem
Neutral class = 41.5% of data. Without intervention, the model will predict "neutral" for ambiguous cases, inflating accuracy but making it useless for real sentiment detection.

### Solution: Class Weights (Recommended — no data discarded)

Compute inverse-frequency weights and pass them to the loss function:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Labels in the training set
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([-1, 0, 1]),
    y=train_df['label'].values
)
# Result will be approximately: [-1: 1.64, 0: 0.80, 1: 0.87]
# This penalises wrong predictions on the minority class (negative) more heavily

weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
```

**Label encoding for CrossEntropyLoss** (requires 0-indexed classes):
```
-1  →  0   (negative)
 0  →  1   (neutral)
 1  →  2   (positive)
```
Apply this mapping before training, reverse it after inference.

### Alternative: Undersampling Neutral
If class weights still don't help enough, cap neutral rows to ~20,000 in training set:
```python
neutral = train_df[train_df['label'] == 0].sample(n=20000, random_state=42)
non_neutral = train_df[train_df['label'] != 0]
train_balanced = pd.concat([neutral, non_neutral]).sample(frac=1, random_state=42)
```

---

## 6. Hyperparameters

```python
MODEL_NAME         = "distilbert-base-uncased"
NUM_LABELS         = 3
MAX_LENGTH         = 128          # max token length — covers 95%+ of tweets/comments
BATCH_SIZE         = 16           # T4 GPU: 16 is safe; reduce to 8 if OOM
GRADIENT_ACCUM     = 2            # effective batch = 16 * 2 = 32
EPOCHS             = 3            # 3 epochs is standard for fine-tuning BERT
LEARNING_RATE      = 2e-5         # standard BERT fine-tuning LR (range: 1e-5 to 5e-5)
WARMUP_RATIO       = 0.1          # 10% of total steps for linear warmup
WEIGHT_DECAY       = 0.01         # L2 regularisation on all params except bias/LayerNorm
MAX_TRAIN_SAMPLES  = None         # set to e.g. 20000 for a quick test run first
SEED               = 42
```

**Why these values:**
- `MAX_LENGTH=128`: 99% of tweets are under 100 tokens; going to 256 doubles memory for marginal gain
- `LR=2e-5`: Too high (>5e-5) causes catastrophic forgetting of pretrained weights; too low (<1e-5) converges poorly
- `WARMUP_RATIO=0.1`: Prevents large gradient updates in early steps from destroying pretrained representations
- `GRADIENT_ACCUM=2`: Simulates larger batch size without needing more VRAM

---

## 7. Training Pipeline

### Step 1: Load and Encode Data
```python
import pandas as pd
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Label encoding: -1→0, 0→1, 1→2
label_map = {-1: 0, 0: 1, 1: 2}
reverse_label_map = {0: -1, 1: 0, 2: 1}

train_df = pd.read_csv("bert_train.csv")
test_df  = pd.read_csv("bert_test.csv")

train_df['encoded_label'] = train_df['label'].map(label_map)
test_df['encoded_label']  = test_df['label'].map(label_map)
```

### Step 2: PyTorch Dataset Class
```python
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }
```

### Step 3: Model Initialisation
```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)
model.to(device)
```

### Step 4: Optimizer and Scheduler
```python
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUM)) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### Step 5: Training Loop
```python
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

training_history = []

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        loss = loss_fn(logits, labels) / GRADIENT_ACCUM
        loss.backward()
        total_loss += loss.item() * GRADIENT_ACCUM

        if (step + 1) % GRADIENT_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_train_loss = total_loss / len(train_loader)

    # --- Evaluation ---
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc       = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average='macro')

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    training_history.append({
        "epoch": epoch + 1,
        "train_loss": round(avg_train_loss, 4),
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4)
    })
```

---

## 8. Evaluation Metrics

Always report ALL of these — accuracy alone is misleading on imbalanced data:

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Map predictions back to original labels (-1, 0, 1)
original_preds  = [reverse_label_map[p] for p in all_preds]
original_labels = [reverse_label_map[l] for l in all_labels]

# Full classification report
print(classification_report(
    original_labels, original_preds,
    target_names=["negative (-1)", "neutral (0)", "positive (1)"]
))

# Confusion matrix
cm = confusion_matrix(original_labels, original_preds, labels=[-1, 0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"])
plt.title("DistilBERT — Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("confusion_matrix_bert.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Target performance (realistic for this dataset):**
| Metric | Target |
|--------|--------|
| Overall Accuracy | ≥ 74% |
| Macro F1 | ≥ 0.70 |
| Negative F1 | ≥ 0.62 |
| Neutral F1 | ≥ 0.72 |
| Positive F1 | ≥ 0.76 |

---

## 9. Save Model, Tokenizer, and Metrics

```python
import json

SAVE_DIR = "/content/bert_sentiment_model"

# Save model and tokenizer (HuggingFace format)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Save training metrics
with open(f"{SAVE_DIR}/training_metrics.json", "w") as f:
    json.dump({
        "model": "distilbert-base-uncased",
        "num_labels": 3,
        "label_map": {"negative": -1, "neutral": 0, "positive": 1},
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "history": training_history,
        "final_accuracy": training_history[-1]["test_accuracy"],
        "final_macro_f1": training_history[-1]["test_macro_f1"]
    }, indent=2)

# Save confusion matrix image
import shutil
shutil.copy("confusion_matrix_bert.png", SAVE_DIR)

print("Saved files:")
import os
for f in os.listdir(SAVE_DIR):
    size = os.path.getsize(f"{SAVE_DIR}/{f}") / 1024**2
    print(f"  {f}  ({size:.1f} MB)")
```

### Expected files after saving:
```
bert_sentiment_model/
├── config.json                 (~1 KB)
├── pytorch_model.bin           (~250 MB)   ← main weights
├── tokenizer_config.json       (~1 KB)
├── tokenizer.json              (~2 MB)
├── vocab.txt                   (~200 KB)
├── special_tokens_map.json     (~1 KB)
├── training_metrics.json       (~2 KB)
└── confusion_matrix_bert.png   (~100 KB)
```
**Total zip size: ~260 MB**

---

## 10. Download from Colab

```python
# Zip the model folder
import shutil
shutil.make_archive("/content/bert_sentiment_model", "zip", "/content/bert_sentiment_model")

# Download the zip
from google.colab import files
files.download("/content/bert_sentiment_model.zip")
```

**After downloading:**
1. Transfer `bert_sentiment_model.zip` to the AWS instance
2. Unzip into `BERT_DATASET/bert_model/`
3. The path `BERT_DATASET/bert_model/` is where Flask will load the model from

---

## 11. Inference (How Flask Will Use This Model)

After the model is placed in `BERT_DATASET/bert_model/`, inference works like this:

```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load once at startup
tokenizer = DistilBertTokenizerFast.from_pretrained("BERT_DATASET/bert_model")
model     = DistilBertForSequenceClassification.from_pretrained("BERT_DATASET/bert_model")
model.eval()

# Label decoding
DECODE = {0: -1, 1: 0, 2: 1}
SENTIMENT = {-1: "negative", 0: "neutral", 1: "positive"}

def predict_bert(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs     = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_idx  = int(torch.argmax(logits, dim=1))
    label     = DECODE[pred_idx]
    return {
        "label":      label,
        "sentiment":  SENTIMENT[label],
        "confidence": round(max(probs), 4),
        "scores":     {
            "negative": round(probs[0], 4),
            "neutral":  round(probs[1], 4),
            "positive": round(probs[2], 4)
        }
    }
```

---

## 12. MLflow Logging (After Model Returns to AWS Instance)

When the trained model is back on the instance, log the following to MLflow:

```python
import mlflow

with mlflow.start_run(run_name="distilbert-sentiment"):
    # Log hyperparameters
    mlflow.log_params({
        "model":           "distilbert-base-uncased",
        "max_length":      128,
        "epochs":          3,
        "batch_size":      16,
        "learning_rate":   2e-5,
        "train_rows":      77120,
        "test_rows":       19281,
        "class_weighting": True,
    })

    # Log final metrics
    mlflow.log_metrics({
        "test_accuracy": <final_accuracy_from_training_metrics.json>,
        "test_macro_f1": <final_macro_f1_from_training_metrics.json>,
    })

    # Log per-epoch history
    for entry in training_history:
        mlflow.log_metrics({
            "train_loss":     entry["train_loss"],
            "epoch_accuracy": entry["test_accuracy"],
            "epoch_macro_f1": entry["test_macro_f1"],
        }, step=entry["epoch"])

    # Log confusion matrix image
    mlflow.log_artifact("BERT_DATASET/bert_model/confusion_matrix_bert.png")

    # Log entire model folder as artifact
    mlflow.log_artifacts("BERT_DATASET/bert_model", artifact_path="bert_model")

    # Register model
    mlflow.register_model(
        f"runs:/<run_id>/bert_model",
        "bert-sentiment"
    )
```

---

## 13. Estimated Training Time on Colab Free Tier (T4 GPU)

| Configuration | Estimated Time |
|---------------|---------------|
| Full 77k rows, 3 epochs, batch=16 | 90–120 min |
| Capped 30k rows, 3 epochs, batch=16 | 35–45 min |
| Capped 20k rows, 2 epochs, batch=16 | 20–25 min |

**Recommendation for first run:** Cap at 20,000 training samples to verify everything works end-to-end before committing to a full 90-minute run. Once confirmed, run full training.

---

## 14. Quick Sanity Check Before Full Training

Before running full training, run this quick check with 500 samples to verify the pipeline works:

```python
# Quick pipeline test — ~2 minutes
QUICK_TEST = True

if QUICK_TEST:
    train_df = train_df.sample(n=500, random_state=42)
    test_df  = test_df.sample(n=100, random_state=42)
    EPOCHS = 1
    print("Running quick test with 500 samples, 1 epoch")
```

---

## 15. Folder Structure After Everything Is Done

```
BERT_DATASET/
├── tweeteval_sentiment.csv       ← raw TweetEval download (59,869 rows)
├── merged_dataset.csv            ← Reddit + TweetEval combined (96,401 rows)
├── bert_train.csv                ← 80% split (77,120 rows)
├── bert_test.csv                 ← 20% split (19,281 rows)
├── prepare_bert_data.py          ← script that created the splits
├── download_tweeteval.py         ← script that downloaded TweetEval
├── plan_bert.md                  ← this file
└── bert_model/                   ← place trained model here after Colab
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    ├── special_tokens_map.json
    ├── training_metrics.json
    └── confusion_matrix_bert.png
```
