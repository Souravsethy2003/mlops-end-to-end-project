import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_reddit_data() -> pd.DataFrame:
    """Load original Reddit sentiment dataset."""
    url = 'https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
    try:
        df = pd.read_csv(url)
        df = df[['clean_comment', 'category']].copy()
        df['source'] = 'reddit'
        logger.debug('Reddit data loaded: %d rows', len(df))
        return df
    except Exception as e:
        logger.error('Failed to load Reddit data: %s', e)
        raise


def load_sentiment140(sample_size: int = 40000) -> pd.DataFrame:
    """Load TweetEval sentiment dataset from HuggingFace (replacement for Sentiment140).
    Original labels: 0=negative, 1=neutral, 2=positive.
    Mapped to: -1, 0, 1.
    """
    try:
        from datasets import load_dataset
        # Load all splits
        frames = []
        for split in ['train', 'validation', 'test']:
            ds = load_dataset('tweet_eval', 'sentiment', split=split)
            frames.append(ds.to_pandas()[['text', 'label']])
        df = pd.concat(frames, ignore_index=True)
        df.columns = ['clean_comment', 'category']
        # Map: 0->-1 (negative), 1->0 (neutral), 2->1 (positive)
        label_map = {0: -1, 1: 0, 2: 1}
        df['category'] = df['category'].map(label_map)
        df = df.dropna(subset=['category'])
        df['category'] = df['category'].astype(int)
        df['source'] = 'tweet_eval'

        # Balanced sample
        if sample_size and sample_size < len(df):
            per_class = sample_size // df['category'].nunique()
            sampled = []
            for cat in df['category'].unique():
                subset = df[df['category'] == cat]
                n = min(per_class, len(subset))
                sampled.append(subset.sample(n=n, random_state=42))
            df = pd.concat(sampled, ignore_index=True)

        logger.debug('TweetEval sentiment loaded: %d rows', len(df))
        return df
    except Exception as e:
        logger.error('Failed to load TweetEval: %s', e)
        return pd.DataFrame(columns=['clean_comment', 'category', 'source'])


def load_sst2() -> pd.DataFrame:
    """Load Stanford Sentiment Treebank v2 from HuggingFace.
    Labels: 0=negative, 1=positive. Mapped to -1, 1.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset('stanfordnlp/sst2')
        # Combine train and validation splits
        frames = []
        for split in ['train', 'validation']:
            if split in ds:
                split_df = ds[split].to_pandas()[['sentence', 'label']].copy()
                frames.append(split_df)
        df = pd.concat(frames, ignore_index=True)
        df.columns = ['clean_comment', 'category']
        # Map: 0->-1 (negative), 1->1 (positive)
        df['category'] = df['category'].map({0: -1, 1: 1})
        df = df.dropna(subset=['category'])
        df['category'] = df['category'].astype(int)
        df['source'] = 'sst2'
        logger.debug('SST-2 loaded: %d rows', len(df))
        return df
    except Exception as e:
        logger.error('Failed to load SST-2: %s', e)
        return pd.DataFrame(columns=['clean_comment', 'category', 'source'])


def load_goemotions() -> pd.DataFrame:
    """Load GoEmotions dataset and map 28 emotion labels to 3-class sentiment.
    Positive emotions -> 1, Negative emotions -> -1, Neutral -> 0.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset('google-research-datasets/go_emotions', 'simplified')
        df = ds['train'].to_pandas()[['text', 'labels']].copy()

        # Emotion index mappings (from GoEmotions label list)
        # 0:admiration, 1:amusement, 2:anger, 3:annoyance, 4:approval, 5:caring,
        # 6:confusion, 7:curiosity, 8:desire, 9:disappointment, 10:disapproval,
        # 11:disgust, 12:embarrassment, 13:excitement, 14:fear, 15:gratitude,
        # 16:grief, 17:joy, 18:love, 19:nervousness, 20:optimism, 21:pride,
        # 22:realization, 23:relief, 24:remorse, 25:sadness, 26:surprise, 27:neutral
        positive_ids = {0, 1, 4, 5, 8, 13, 15, 17, 18, 20, 21, 23}  # admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
        negative_ids = {2, 3, 9, 10, 11, 12, 14, 16, 19, 24, 25}     # anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
        neutral_ids = {6, 7, 22, 26, 27}                              # confusion, curiosity, realization, surprise, neutral

        def map_emotion_to_sentiment(label_list):
            pos = sum(1 for l in label_list if l in positive_ids)
            neg = sum(1 for l in label_list if l in negative_ids)
            if pos > neg:
                return 1
            elif neg > pos:
                return -1
            return 0

        df['category'] = df['labels'].apply(map_emotion_to_sentiment)
        df = df.rename(columns={'text': 'clean_comment'})[['clean_comment', 'category']]
        df['source'] = 'goemotions'
        logger.debug('GoEmotions loaded: %d rows', len(df))
        return df
    except Exception as e:
        logger.error('Failed to load GoEmotions: %s', e)
        return pd.DataFrame(columns=['clean_comment', 'category', 'source'])


def load_youtube_scraped(data_dir: str) -> pd.DataFrame:
    """Load self-scraped YouTube comments from data/external/youtube_comments.csv."""
    path = os.path.join(data_dir, 'external', 'youtube_comments.csv')
    try:
        if not os.path.exists(path):
            logger.debug('No YouTube scraped data found at %s, skipping', path)
            return pd.DataFrame(columns=['clean_comment', 'category', 'source'])
        df = pd.read_csv(path)
        df = df[['clean_comment', 'category']].copy()
        df['source'] = 'youtube'
        logger.debug('YouTube scraped data loaded: %d rows', len(df))
        return df
    except Exception as e:
        logger.error('Failed to load YouTube scraped data: %s', e)
        return pd.DataFrame(columns=['clean_comment', 'category', 'source'])


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        df = df.dropna(subset=['clean_comment', 'category'])
        df = df.drop_duplicates(subset=['clean_comment'])
        df = df[df['clean_comment'].str.strip() != '']
        df['category'] = df['category'].astype(int)
        logger.debug('Data preprocessing completed: %d rows remaining', len(df))
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def balance_classes(df: pd.DataFrame, strategy: str = 'undersample') -> pd.DataFrame:
    """Balance class distribution via undersampling or oversampling."""
    if strategy == 'none':
        return df

    counts = df['category'].value_counts()
    logger.debug('Class distribution before balancing: %s', counts.to_dict())

    if strategy == 'undersample':
        min_count = counts.min()
        balanced = []
        for cat in counts.index:
            subset = df[df['category'] == cat]
            balanced.append(subset.sample(n=min_count, random_state=42))
        df = pd.concat(balanced, ignore_index=True)
    elif strategy == 'oversample':
        max_count = counts.max()
        balanced = []
        for cat in counts.index:
            subset = df[df['category'] == cat]
            balanced.append(subset.sample(n=max_count, replace=True, random_state=42))
        df = pd.concat(balanced, ignore_index=True)

    logger.debug('Class distribution after balancing: %s', df['category'].value_counts().to_dict())
    return df


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        ingestion_params = params['data_ingestion']
        test_size = ingestion_params['test_size']

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')

        # Load all data sources
        frames = []

        # 1. Original Reddit data (always loaded)
        logger.info('Loading Reddit data...')
        frames.append(load_reddit_data())

        # 2. Sentiment140
        if ingestion_params.get('use_sentiment140', False):
            logger.info('Loading Sentiment140...')
            sample = ingestion_params.get('sentiment140_sample', 40000)
            frames.append(load_sentiment140(sample_size=sample))

        # 3. SST-2
        if ingestion_params.get('use_sst2', False):
            logger.info('Loading SST-2...')
            frames.append(load_sst2())

        # 4. GoEmotions
        if ingestion_params.get('use_goemotions', False):
            logger.info('Loading GoEmotions...')
            frames.append(load_goemotions())

        # 5. YouTube scraped data
        if ingestion_params.get('use_youtube_scraped', False):
            logger.info('Loading YouTube scraped data...')
            frames.append(load_youtube_scraped(data_dir))

        # Merge all sources
        df = pd.concat(frames, ignore_index=True)
        logger.info('Total rows after merge: %d', len(df))
        logger.info('Sources: %s', df['source'].value_counts().to_dict())

        # Preprocess (dedup, remove empty, etc.)
        final_df = preprocess_data(df)

        # Balance classes
        balance_strategy = ingestion_params.get('balance_strategy', 'undersample')
        final_df = balance_classes(final_df, strategy=balance_strategy)

        logger.info('Final dataset: %d rows', len(final_df))
        logger.info('Final class distribution: %s', final_df['category'].value_counts().to_dict())
        logger.info('Final source distribution: %s', final_df['source'].value_counts().to_dict())

        # Split
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42, stratify=final_df['category']
        )

        # Save
        save_data(train_data, test_data, data_path=data_dir)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
