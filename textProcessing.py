from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pickle

from load_dataset import *


def serializeClassifier(classifier, modelFilePath):
    save_classifier = open(modelFilePath, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


if __name__ == "__main__":

    # Name of the classifier to use in learning process and its path of serialization
    classifierType = "svm"
    modelFilePath = "models/" + classifierType + ".pickle"

    # Defining categories of the classes to use in model
    categories = {'Positive' : 'pos', 'Negative' : 'neg'}

    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    dataset_labels, dataset = load_dataset('Twitter')

    # Using the vector count class to count the terms and tokens in the data-set
    count_vect = CountVectorizer(encoding='latin-1')
    X_train_counts = count_vect.fit_transform(dataset.values())

    # Using the TF-IDS transformation as a feature of the data-set
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # An SVM classifier to use for the learning process
    classifier = SGDClassifier(loss='hinge', penalty='l2',
                  alpha=1e-3, random_state=42,
                  max_iter=5, tol=None)

    # Fitting the classifier using the data-set
    classifier.fit(X_train_tfidf, dataset_labels)

    # Serializing the classifier
    serializeClassifier(classifier, modelFilePath)

    # Testing
    docs_new = ['اسال الله في علاه ان يحفظكم بما يحفظ به عباده الصالحون']

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = classifier.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, categories[category]))


    # Evaluation

def build_pipeline():
    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    dataset_labels, dataset = load_dataset('Twitter')


    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None)),
                         ])
    text_clf.fit(dataset, dataset_labels)

    predicted = text_clf.predict(dataset)
    np.mean(predicted == dataset_labels)

    print(metrics.classification_report(dataset_labels, predicted,
                                        target_names=categories))

    metrics.confusion_matrix(dataset_labels, predicted)