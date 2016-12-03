from __future__ import print_function
from time import time

from nltk.corpus import stopwords as sw

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

n_samples = 2000
n_features = 10000
n_topics = 1
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ,".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def load_topics(data_samples):
    # Use tf-idf features for NMF.
    #print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1,ngram_range=(2,2),token_pattern=r'\b\w+\b',
                                       max_features=n_features, stop_words='english',analyzer ='word',
                                       )

    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples.values.astype('U'))
    #print('Stop words',tfidf_vectorizer.get_feature_names())

    #print("done in %0.3fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    #print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1,ngram_range=(2,2),token_pattern=r'\b\w+\b',
                                    max_features=n_features,
                                    stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples.values.astype('U'))
    #print("done in %0.3fs." % (time() - t0))

    # Fit the NMF model
    #print("Fitting the NMF model with tf-idf features, "          "n_samples=%d and n_features=%d..."    % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    #print("done in %0.3fs." % (time() - t0))

    print("Topics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    #print("Fitting LDA models with tf features, ",       "n_samples=%d and n_features=%d...",          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    print("Topics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
#  only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = pd.read_csv("test.csv",header=0)
dataset = dataset[dataset['Label'].str.len() > 0]
#dataset = pd.read_csv("train.csv",header=0)
print('Romantic')
dataset_love=dataset[dataset['Label'] == 'Romantic']
n_samples=dataset_love.shape[0]
data_samples = dataset_love.Lyrics[:n_samples]
print(dataset_love.shape)
#print("done in %0.3fs." % (time() - t0))
load_topics(data_samples)

print('Sad')
dataset_love=dataset[dataset['Label'] == 'Sad']
n_samples=dataset_love.shape[0]
data_samples = dataset_love.Lyrics[:n_samples]
print(dataset_love.shape)
#print("done in %0.3fs." % (time() - t0))
load_topics(data_samples)

print('Party')
dataset_love=dataset[dataset['Label'] == 'Party']
n_samples=dataset_love.shape[0]
data_samples = dataset_love.Lyrics[:n_samples]
print(dataset_love.shape)
#print("done in %0.3fs." % (time() - t0))
load_topics(data_samples)

print('Happy')
dataset_love=dataset[dataset['Label'] == 'Happy']
n_samples=dataset_love.shape[0]
data_samples = dataset_love.Lyrics[:n_samples]
print(dataset_love.shape)
#print("done in %0.3fs." % (time() - t0))
load_topics(data_samples)

print('Betray')
dataset_love=dataset[dataset['Label'] == 'Betray']
n_samples=dataset_love.shape[0]
data_samples = dataset_love.Lyrics[:n_samples]
print(dataset_love.shape)
#print("done in %0.3fs." % (time() - t0))
load_topics(data_samples)
