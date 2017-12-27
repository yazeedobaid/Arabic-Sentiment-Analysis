from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#-----------------------------------------------------------------------------------------------------------------------
# Extracting the TF-IDF (Term-frequency Inverse-Term-Frequency) features from the data-set.
#
#   PARAMETERS:
#       - dataset: dictionary of the data-set
#
#   RETURNS:
#       - count_vect: count vector of the terms in data-set
#       - X_train_tfidf: the tf-idf list of the terms in data-set
#       - tfidf_transformer: the tf-idf transformer object
#-----------------------------------------------------------------------------------------------------------------------
def tf_idf_features(dataset=None):
    # Check if the data-set and its labels are provided
    if dataset == None:
        print('No data-set provided!')
        exit(1)

    # Using the vector count class to count the terms and tokens in the data-set
    count_vect = CountVectorizer(encoding='latin-1', ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(dataset.values())

    # Using the TF-IDS transformation as a feature of the data-set
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return count_vect, X_train_tfidf, tfidf_transformer