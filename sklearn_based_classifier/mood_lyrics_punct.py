import os
import time
import string
import pickle

from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import libsvm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts


output_logs=open(r'data/output.txt','w')
def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        #self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)



@timeit
def build_and_evaluate(X, y, classifier=SGDClassifier, outpath=None, verbose=True):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.
    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.
    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.
    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """

    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()

        #print ('TfidfVectorizer',TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False))

        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])

        model.fit(X, y)
        print('model created')
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model, secs = build(classifier, X_train, y_train)

    if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    #Evaluating the model
    print(clsr(y_test, y_pred, target_names=labels.classes_))

    if verbose: print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def show_most_informative_features(model, text=None, n=20):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    try:
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']

        # Check to make sure that we can perform this computation
        if not hasattr(classifier, 'coef_'):
            raise TypeError(
                "Cannot compute most informative features on {} model.".format(
                    classifier.__class__.__name__
                )
            )

        if text is not None:
            # Compute the coefficients for the text
            tvec = model.transform([text]).toarray()
        else:
            # Otherwise simply use the coefficients
            tvec = classifier.coef_

        # Zip the feature names with the coefs and sort
        coefs = sorted(
            zip(tvec[0], vectorizer.get_feature_names()),
            key=itemgetter(0), reverse=True
        )

        topn  = zip(coefs[:n], coefs[:-(n+1):-1])

        # Create the output string to return
        output = []

        # If text, add the predicted value to the output.
        if text is not None:
            output.append("\"{}\"".format(text))
            output.append("Classified as: {}".format(model.predict([text])))
            output.append("")

        # Create two columns with most negative and most positive features.
        for (cp, fnp), (cn, fnn) in topn:
            output.append(
                "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
            )

        return "\n".join(output)
    except:
        return "\n TypeError: Cannot compute most informative features on THIS Classifier model."

def movie_example():
    PATH = "model.pickle"

    if not os.path.exists(PATH):
        # Time to build the model
        from nltk.corpus import movie_reviews as reviews
        print(reviews.columns)
        X = [reviews.raw(fileid) for fileid in reviews.fileids()]
        y = [reviews.categories(fileid)[0] for fileid in reviews.fileids()]
        print(X[0],'\n\n')
        print(y[0])
        #X,y=X[:10],y[:10]
        model = build_and_evaluate(X,y, outpath=PATH)

    else:
        with open(PATH, 'rb') as f:
            model = pickle.load(f)

    print(show_most_informative_features(model))

def get_song_attributes(train_file):
    rank, song, artist, year, lyrics, source, label = [], [], [], [], [], [], []
    count = 0
    with open(train_file, "r",encoding='latin1') as lines:
        #data = file.read()
        #lines = data.split('\n')
        for line in lines:
            row = line.split(',')
            if count == 0:
                print ("skip header")
                count+=1
            else:
                if len(row) == 8 and row[6] != '':
                #if len(row) == 8 and row[6]!='' and row[7] in users and !(row[4].):
                    rank.append(row[0])
                    song.append(row[1])
                    artist.append(row[2])
                    year.append(row[3])
                    lyrics.append(row[4])
                    source.append(row[5])
                    label.append(row[6])
    return rank, song, artist, year, lyrics, source, label

def mood_lyrics():
    PATH = "mood_model.pickle"
    #train_file = 'data/Cleaned-Data/anusha_cleaned_lyrics.csv'
    train_file = 'data/Cleaned-Data/PositiveNegative.csv'
    rank, song, artist, year, lyrics, source, label=get_song_attributes(train_file)
    X = lyrics
    y = label


    print('MultinomialNB')
    model = build_and_evaluate(X,y, outpath=PATH,classifier = MultinomialNB)
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    print(show_most_informative_features(model))

    print('SGDClassifier')
    model = build_and_evaluate(X,y, outpath=PATH,classifier = SGDClassifier)
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    print(show_most_informative_features(model))

    print('DecisionTreeClassifier')
    model = build_and_evaluate(X,y, outpath=PATH,classifier = DecisionTreeClassifier)
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    print(show_most_informative_features(model))

    print('AdaBoostClassifier')
    model = build_and_evaluate(X, y, outpath=PATH, classifier=AdaBoostClassifier)
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    print(show_most_informative_features(model))

    print('KNeighborsClassifier')
    model = build_and_evaluate(X, y, outpath=PATH, classifier=KNeighborsClassifier)
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    print(show_most_informative_features(model))

mood_lyrics()