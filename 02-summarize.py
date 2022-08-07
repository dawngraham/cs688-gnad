#!/usr/bin/env python3

import logging
import sys

import pandas as pd
import regex as re
import unidecode
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from summarizer import Summarizer
from tqdm import tqdm

tqdm.pandas()

TEST_FILE = './02-summarize-test.txt'
SRC_FILE = './data/campaigns.parquet.gzip'
TGT_FILE = './data/campaigns_summaries.parquet.gzip'

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s',
                                 datefmt='%m/%d/%Y %I:%M:%S%p')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# Raise level for other loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != __name__:
        logging.getLogger(log_name).setLevel(logging.ERROR)


def get_summary(df):
    def summarize(body):
        model = Summarizer()
        result = model(body, num_sentences=3)
        return result

    logger.info('Getting summaries...')
    df['summary'] = df.progress_apply(lambda x: summarize(x['details']), axis=1)
    return df


def get_preprocessed_text(df):
    def preprocess(body):
        # Get rid of accents
        unaccented = unidecode.unidecode(body)

        # Get rid of punctuation & numbers
        letters_only = re.sub("[^a-zA-Z]", " ", unaccented)

        # Get all lowercase words
        words = letters_only.lower().split()

        # Remove stop words
        stops = set(stopwords.words('english'))
        meaningful_words = [w for w in words if w not in stops]

        # Instantiate and run Lemmatizer
        lemmatizer = WordNetLemmatizer()
        tokens_lem = [lemmatizer.lemmatize(i) for i in meaningful_words]

        # Join back into string
        result = " ".join(tokens_lem)

        # Join into string and return the result.
        return result

    logger.info('Getting preprocessed text...')
    df['details_clean'] = df.progress_apply(lambda x: preprocess(x['details']), axis=1)
    df['summary_clean'] = df.progress_apply(lambda x: preprocess(x['summary']), axis=1)
    return df


def get_ngram1(df):
    def ngram1(body):
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords = model.extract_keywords(body, top_n=5)
        return keywords

    logger.info('Getting ngram1...')
    df['ngram1_details'] = df.progress_apply(lambda x: ngram1(x['details_clean']), axis=1)
    df['ngram1_summary'] = df.progress_apply(lambda x: ngram1(x['summary_clean']), axis=1)
    return df


def get_ngram2(df):
    def ngram2(body):
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords = model.extract_keywords(body, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        return keywords

    logger.info('Getting ngram2...')
    df['ngram2_details'] = df.progress_apply(lambda x: ngram2(x['details_clean']), axis=1)
    df['ngram2_summary'] = df.progress_apply(lambda x: ngram2(x['summary_clean']), axis=1)
    return df


def get_ngram_maxsum(df):
    def ngram_maxsum(body):
        """Max Sum Similarity
        To diversify the results, we take the 2 x top_n most similar words/phrases to the document.
        Then, we take all top_n combinations from the 2 x top_n words and extract the combination
        that are the least similar to each other by cosine similarity.
        """
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords = model.extract_keywords(body, keyphrase_ngram_range=(3, 3), stop_words='english', use_maxsum=True,
                                          nr_candidates=20, top_n=5)
        return keywords

    logger.info('Getting ngram maxsum...')
    df['maxsum_details'] = df.progress_apply(lambda x: ngram_maxsum(x['details_clean']), axis=1)
    df['maxsum_summary'] = df.progress_apply(lambda x: ngram_maxsum(x['summary_clean']), axis=1)
    return df


def get_ngram_maxmarginal(df):
    def ngram_maxmarginal(body):
        """Maximal Marginal Relevance
        To diversify the results, create keywords based on cosine similarity.
        """
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords = model.extract_keywords(body, keyphrase_ngram_range=(3, 3), stop_words='english', use_mmr=True,
                                          diversity=0.7, top_n=5)
        return keywords

    logger.info('Getting ngram maxmarginal...')
    df['maxmarginal_details'] = df.progress_apply(lambda x: ngram_maxmarginal(x['details_clean']), axis=1)
    df['maxmarginal_summary'] = df.progress_apply(lambda x: ngram_maxmarginal(x['summary_clean']), axis=1)
    return df


if __name__ == '__main__':
    sys.stdout = open(TEST_FILE, 'w')

    logger.info('Starting test.')
    campaigns = pd.read_parquet(SRC_FILE)
    campaigns = campaigns.head(2)
    campaigns = get_summary(campaigns)
    campaigns = get_preprocessed_text(campaigns)
    campaigns = get_ngram1(campaigns)
    campaigns = get_ngram2(campaigns)
    campaigns = get_ngram_maxsum(campaigns)
    campaigns = get_ngram_maxmarginal(campaigns)

    for campaign in range(len(campaigns)):
        for col in campaigns.columns:
            print(f'{col.upper()}\n{campaigns.loc[campaign, col]}\n')
        print('-------------')

    logger.info('Done with test.')

    logger.info('STARTING ON FULL DATASET')
    campaigns = pd.read_parquet(SRC_FILE)
    campaigns = get_summary(campaigns)
    campaigns = get_preprocessed_text(campaigns)
    campaigns = get_ngram2(campaigns)

    logging.info('Getting separate phrases & values columns...')
    campaigns['details_phrases'] = campaigns.progress_apply(lambda x: [x[0] for x in x['ngram2_details']], axis=1)
    campaigns['details_values'] = campaigns.progress_apply(lambda x: [x[1] for x in x['ngram2_details']], axis=1)
    campaigns['summary_phrases'] = campaigns.progress_apply(lambda x: [x[0] for x in x['ngram2_summary']], axis=1)
    campaigns['summary_values'] = campaigns.progress_apply(lambda x: [x[1] for x in x['ngram2_summary']], axis=1)

    # Save file with url, summary, and keywords only
    logging.info(f'Saving {TGT_FILE}')
    cols = ['url', 'summary', 'details_phrases', 'details_values', 'summary_phrases', 'summary_values']
    campaigns[cols].to_parquet(TGT_FILE, compression='gzip')

    logger.info('Done.')
