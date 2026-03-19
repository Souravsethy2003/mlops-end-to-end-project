"""
TweetEval Sentiment Dataset Downloader & Preprocessor
=======================================================
Downloads the TweetEval 'sentiment' split from HuggingFace,
applies professional preprocessing, and saves a clean CSV with:
  - text        : cleaned tweet text
  - label       : -1 (negative), 0 (neutral), 1 (positive)
  - source      : 'TweetEval'
"""

import os
import re
import logging
import pandas as pd
from datasets import load_dataset

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tweeteval_download")

# ── Label mapping: TweetEval → our 3-class scheme ─────────────────────────────
LABEL_MAP = {0: -1, 1: 0, 2: 1}          # 0=negative→-1, 1=neutral→0, 2=positive→1
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}

# ── Output path ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH    = os.path.join(SCRIPT_DIR, "tweeteval_sentiment.csv")


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """
    Professional tweet cleaning pipeline:
      1. Replace @mentions with a generic token (preserves semantic role)
      2. Strip URLs
      3. Remove RT (retweet) markers
      4. Normalise whitespace & strip edges
      5. Remove residual non-ASCII control characters
    """
    if not isinstance(text, str):
        return ""

    # 1. Anonymise mentions (keep placeholder so BERT knows a person was referenced)
    text = re.sub(r"@\w+", "@user", text)

    # 2. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove RT markers
    text = re.sub(r"\bRT\b", "", text)

    # 4. Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Drop non-printable control characters (keep emojis — they carry sentiment)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text


def is_valid(text: str) -> bool:
    """Drop rows that are empty or too short to carry meaning after cleaning."""
    return isinstance(text, str) and len(text.strip()) >= 5


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading TweetEval 'sentiment' dataset from HuggingFace …")
    splits = ["train", "validation", "test"]
    frames = []

    for split in splits:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split, trust_remote_code=True)
        df = ds.to_pandas()[["text", "label"]].copy()
        df["split"] = split
        frames.append(df)
        log.info("  %-12s → %6d rows", split, len(df))

    raw = pd.concat(frames, ignore_index=True)
    log.info("Total rows downloaded : %d", len(raw))

    # ── Cleaning ────────────────────────────────────────────────────────────────
    log.info("Cleaning tweet text …")
    raw["text"] = raw["text"].apply(clean_tweet)

    # Drop rows too short after cleaning
    before = len(raw)
    raw = raw[raw["text"].apply(is_valid)].copy()
    log.info("Dropped %d rows (empty / too short after cleaning)", before - len(raw))

    # Drop exact duplicates on text (keeps first occurrence)
    before = len(raw)
    raw = raw.drop_duplicates(subset=["text"]).copy()
    log.info("Dropped %d duplicate rows", before - len(raw))

    # ── Label mapping ───────────────────────────────────────────────────────────
    log.info("Mapping labels  0→-1 (negative)  1→0 (neutral)  2→1 (positive) …")
    raw["label"] = raw["label"].map(LABEL_MAP)

    # Sanity-check: drop rows whose original label was outside {0,1,2}
    before = len(raw)
    raw = raw.dropna(subset=["label"]).copy()
    raw["label"] = raw["label"].astype(int)
    dropped_unknown = before - len(raw)
    if dropped_unknown:
        log.warning("Dropped %d rows with unknown label values", dropped_unknown)

    # ── Add source column ───────────────────────────────────────────────────────
    raw["source"] = "TweetEval"

    # ── Final columns (text, label, source) ─────────────────────────────────────
    final = raw[["text", "label", "source"]].reset_index(drop=True)

    # ── Class distribution report ───────────────────────────────────────────────
    log.info("─" * 50)
    log.info("Final dataset shape : %s", final.shape)
    dist = final["label"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        pct = cnt / len(final) * 100
        log.info("  label %+2d  (%s)  →  %6d rows  (%.1f%%)", lbl, LABEL_NAMES[lbl], cnt, pct)
    log.info("─" * 50)

    # ── Save ────────────────────────────────────────────────────────────────────
    final.to_csv(OUT_PATH, index=False)
    log.info("Saved → %s  (%d rows)", OUT_PATH, len(final))
    log.info("Done.")


if __name__ == "__main__":
    main()
