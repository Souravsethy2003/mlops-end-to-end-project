"""
BERT Dataset Preparation
========================
1. Loads Reddit (train + test combined) — keeps originals untouched
2. Loads TweetEval sentiment CSV (already downloaded)
3. Merges both into merged_dataset.csv
4. Stratified 80/20 split → bert_train.csv + bert_test.csv
5. Creates bert_model/ directory placeholder

All three label values: -1 (negative), 0 (neutral), 1 (positive)
Final columns: text | label | source
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bert_data_prep")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

REDDIT_TRAIN  = os.path.join(PROJECT_DIR, "data", "raw", "train.csv")
REDDIT_TEST   = os.path.join(PROJECT_DIR, "data", "raw", "test.csv")
TWEETEVAL_CSV = os.path.join(SCRIPT_DIR, "tweeteval_sentiment.csv")

OUT_MERGED    = os.path.join(SCRIPT_DIR, "merged_dataset.csv")
OUT_TRAIN     = os.path.join(SCRIPT_DIR, "bert_train.csv")
OUT_TEST      = os.path.join(SCRIPT_DIR, "bert_test.csv")
BERT_MODEL_DIR = os.path.join(SCRIPT_DIR, "bert_model")

LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def load_reddit() -> pd.DataFrame:
    """Load Reddit train + test, rename columns to match TweetEval schema."""
    train = pd.read_csv(REDDIT_TRAIN)
    test  = pd.read_csv(REDDIT_TEST)
    df    = pd.concat([train, test], ignore_index=True)
    # Rename to unified schema
    df = df.rename(columns={"clean_comment": "text", "category": "label"})
    df = df[["text", "label", "source"]].copy()
    log.info("Reddit loaded   : %d rows  (train=%d + test=%d)", len(df), len(train), len(test))
    return df


def load_tweeteval() -> pd.DataFrame:
    df = pd.read_csv(TWEETEVAL_CSV)
    log.info("TweetEval loaded: %d rows", len(df))
    return df[["text", "label", "source"]].copy()


def show_dist(df: pd.DataFrame, tag: str):
    log.info("--- %s (%d rows) ---", tag, len(df))
    dist = df["label"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        log.info("  label %+2d (%s) : %6d  (%.1f%%)",
                 lbl, LABEL_NAMES[int(lbl)], cnt, cnt / len(df) * 100)


def main():
    # ── 1. Load ────────────────────────────────────────────────────────────────
    reddit    = load_reddit()
    tweeteval = load_tweeteval()

    # ── 2. Merge (originals are untouched — we only read them) ─────────────────
    merged = pd.concat([reddit, tweeteval], ignore_index=True)
    log.info("After concat    : %d rows", len(merged))

    # Basic cleaning: drop nulls, empty text, ensure label in {-1, 0, 1}
    merged = merged.dropna(subset=["text", "label"])
    merged = merged[merged["text"].str.strip().str.len() >= 5]
    merged = merged[merged["label"].isin([-1, 0, 1])]
    merged["label"] = merged["label"].astype(int)
    merged = merged.drop_duplicates(subset=["text"])
    log.info("After cleaning  : %d rows", len(merged))

    show_dist(merged, "Merged (raw)")

    # ── 3. Save full merged dataset (no balancing — original distribution) ─────
    merged.to_csv(OUT_MERGED, index=False)
    log.info("Saved → %s", OUT_MERGED)

    # ── 4. Stratified 80/20 split ──────────────────────────────────────────────
    train_df, test_df = train_test_split(
        merged,
        test_size=0.20,
        random_state=42,
        stratify=merged["label"],
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    show_dist(train_df, "bert_train.csv")
    show_dist(test_df,  "bert_test.csv")

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST,  index=False)
    log.info("Saved → %s  (%d rows)", OUT_TRAIN, len(train_df))
    log.info("Saved → %s  (%d rows)", OUT_TEST,  len(test_df))

    # ── 5. Create bert_model/ placeholder ─────────────────────────────────────
    os.makedirs(BERT_MODEL_DIR, exist_ok=True)
    readme = os.path.join(BERT_MODEL_DIR, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "Place the trained DistilBERT model files here after downloading from Google Colab.\n"
            "Expected contents after training:\n"
            "  config.json\n"
            "  pytorch_model.bin  (or model.safetensors)\n"
            "  tokenizer_config.json\n"
            "  tokenizer.json\n"
            "  vocab.txt\n"
            "  special_tokens_map.json\n"
            "  training_metrics.json   (loss, accuracy, F1 per epoch)\n"
        )
    log.info("Created → %s/", BERT_MODEL_DIR)
    log.info("All done.")


if __name__ == "__main__":
    main()
