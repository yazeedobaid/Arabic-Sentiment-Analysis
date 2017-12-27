from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pickle
from sklearn.model_selection import cross_val_score
from load_dataset import *
from Stemming import Stemming
from Normalization import Normalization

from stopWords import StopWords


def serializeClassifier(classifier, modelFilePath):
    save_classifier = open(modelFilePath, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


def preprocessing(dataset):
    filtered_dataset = {}
    lang = 'arabic'
    stopwords = StopWords()
    stemming = Stemming()
    normalization = Normalization()
    for exampleKey, exampleValue in dataset.items():
        filtered_sentence = stopwords.removeStopWords(exampleValue, lang)
        filtered_sentence = stemming.stemWord(filtered_sentence)
        filtered_sentence = normalization.normalizeText(filtered_sentence)
        filtered_dataset[exampleKey] = filtered_sentence

    return filtered_dataset


def build_pipeline():
    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    # dataset_labels, dataset = load_dataset('Twitter')

    # Calls the csv_dict_list function, passing the named csv
    dataset_labels, dataset = csv_dict_list("ATT.csv")

    # Building the pipeline processing
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', SGDClassifier(loss='hinge', penalty='l2',
                                     alpha=1e-3, random_state=42,
                                     max_iter=5, tol=None)),
    ])

    # Cross validating the pipeline
    scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                             list(dataset.values()),  # training data
                             dataset_labels,  # training labels
                             cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                             scoring='f1',  # which scoring metric?
                             n_jobs=-1,  # -1 = use all cores = faster
                             )

    print()
    print('Scores of the cross validation folds are: ')
    print(scores)
    print('----------------------------------------------------------------------')
    print('Mean score of the cross validation folds is: ')
    print(scores.mean())

    # Fitting the pipeline on the data-set
    pipeline.fit(dataset, dataset_labels)

    # Using the pipeline to predict new examples. The same training data is used
    predicted = pipeline.predict(dataset)
    np.mean(predicted == dataset_labels)

    # Confusion matrix
    # print()
    # print('Confusion matrix of the testing data (testing data = training data)')
    # print(metrics.classification_report(dataset_labels, predicted,
    #                                     target_names=categories))

    metrics.confusion_matrix(dataset_labels, predicted)


if __name__ == "__main__":

    # Name of the classifier to use in learning process and its path of serialization
    classifierType = "svm"
    modelFilePath = "models/" + classifierType + ".pickle"

    # Defining categories of the classes to use in model
    categories = {'Positive': 'pos', 'Negative': 'neg'}

    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    # dataset_labels, dataset = load_dataset('Twitter')

    # Calls the csv_dict_list function, passing the named csv
    dataset_labels, dataset = csv_dict_list("ATT.csv")

    # Preprocessing of data-set
    dataset = preprocessing(dataset)

    # Using the vector count class to count the terms and tokens in the data-set
    count_vect = CountVectorizer(encoding='latin-1', ngram_range=(1, 2))
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
    print('Testing a new data example ...')
    docs_new = ['الحياة صعبة شباب']

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = classifier.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        if category == 1 or category == -1:
            print('%r => %s' % (doc, categories['Positive' if category else 'Negative']))
        else:
            print('%r => %s' % (doc, categories[category]))

    build_pipeline()
