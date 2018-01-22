import time

import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from preprocessing.Normalization import Normalization
from preprocessing.Stemming import Stemming
from preprocessing.stopWords import StopWords

from utilities.feature_extraction import *
from utilities.load_dataset import *

from sklearn.naive_bayes import MultinomialNB

from sentiment_analysis import *


#-----------------------------------------------------------------------------------------------------------------------
# Rre-process the data-set, pre-processing includes:
#   1. Stop word removal using NLTK pre-defined Arabic stop word list.
#   2. Sentence tokenization into words.
#   3. Stemming of each word using NLTK isri stemmer.
#   4. Normalization using pyArabic module.
#
#   PARAMETERS:
#       - dataset: dictionary of data-set samples
#
#   RETURNS:
#       - filtered data-set: data-set after pre-processing
#-----------------------------------------------------------------------------------------------------------------------
def preprocessing(dataset):
    lang = 'arabic'
    stopwords = StopWords()
    stemming = Stemming()
    normalization = Normalization()

    if isinstance(dataset, dict):
        filtered_dataset = dict()
        for exampleKey, exampleValue in dataset.items():
            filtered_sentence = stopwords.removeStopWords(exampleValue, lang)
            filtered_sentence = stemming.stemWord(filtered_sentence)
            filtered_sentence = normalization.normalizeText(filtered_sentence)
            filtered_dataset[exampleKey] = filtered_sentence
        return filtered_dataset
    else:
        filtered_example = ''
        filtered_sentence = stopwords.removeStopWords(dataset, lang)
        filtered_sentence = stemming.stemWord(filtered_sentence)
        filtered_sentence = normalization.normalizeText(filtered_sentence)
        filtered_example = filtered_sentence
        return filtered_example


#-----------------------------------------------------------------------------------------------------------------------
# Build a processing pipeline and evaluate the pipeline classifier using cross validation
# technique.
#
#   PARAMETERS:
#       - dataset: a dictionary of the data-set samples
#       - dataset_labels: labels of each data-set example
#
#   RETURNS:
#       - No return value!
#-----------------------------------------------------------------------------------------------------------------------
def build_pipeline(dataset, dataset_labels, scoring_metric):
    # Building the pipeline processing
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
    ])

    # USE FOR naive bias classifier
    # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    # USE FOR SVM SGD classifier
    # SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

    # Cross validating the pipeline
    scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                             list(dataset.values()),  # training data
                             dataset_labels,  # training labels
                             cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                             scoring=scoring_metric,  # which scoring metric?
                             n_jobs=1,  # -1 = use all cores = faster
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



# -----------------------------------------------------------------------------------------------------------------------
# The main method of the script. The script train and test the classifier on a new input data
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Name of the classifier to use in learning process and its path of serialization
    classifierType = "svm"
    modelFilePath = "models/" + classifierType + ".pickle"

    # Defining categories of the classes to use in model
    categories = {'Positive': 'pos', 'Negative': 'neg'}

    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    # dataset_labels, dataset = load_dataset('datasets/Twitter')

    # Calls the csv_dict_list function, passing the named csv
    dataset_labels, dataset = csv_dict_list("datasets/ATT.csv")

    # Preprocessing of data-set
    dataset = preprocessing(dataset)

    # feature extraction, tf-idf transformation
    count_vect, X_train_tfidf, tfidf_transformer = tf_idf_features(dataset)

    # an object from sentiment analysis module to use in training and testing
    sentiment_analysis = sentiment_analysis()

    # train a classifier
    print('Training a classifier is in progress ...')
    classifier = sentiment_analysis.sentiment_analysis_train(X_train_tfidf, dataset_labels, classifierType, modelFilePath)

    print('Training done')
    print('-------------------------------------------------------------')
    # loading the classifier. Un-commit if a classifier already exists
    #classifier = readSerializedClassifier(modelFilePath)

    # testing a new input example
    print('Testing a new data example ...')
    input_text = ['الحياة صعبة شباب']

    filtered_input_text = list()
    filtered_input_text.append(preprocessing(''.join(input_text)))

    sentiment_analysis.sentiment_analysis_test(filtered_input_text, classifier, count_vect, X_train_tfidf, tfidf_transformer, categories)


    # evaluation of the classifier
    # dataset_labels, dataset = load_dataset('datasets/Twitter')

    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    dataset_labels, dataset = csv_dict_list("datasets/ATT.csv")
    build_pipeline(dataset, dataset_labels, 'f1')


def all_datasets_results():
    # the metric to use in the evaluation of the cross validation on the data-set
    scoring_metric = 'f1'

    # Loading the data-set. The data-set is loaded as a dictionary with each
    # element contains the content of the example file
    # build the pipeline and get results

    # testing the model on Twitter data-set
    print('***********************************************************************************************************')
    print('Testing the model on Twitter Tweets data-set')
    dataset_labels, dataset = load_dataset('datasets/Twitter')
    build_pipeline(dataset, dataset_labels, scoring_metric)

    # testing the model on Attraction Reviews data-set
    print('***********************************************************************************************************')
    print('Testing the model on Attraction Reviews data-set')
    dataset_labels, dataset = csv_dict_list('datasets/ATT.csv')
    build_pipeline(dataset, dataset_labels, scoring_metric)

    # testing the model on Hotel Reviews data-set
    dataset_labels, dataset = csv_dict_list('datasets/HTL.csv')
    build_pipeline(dataset, dataset_labels, scoring_metric)

    # testing the model on Movie Reviews data-set
    print('***********************************************************************************************************')
    print('Testing the model on Movie Reviews data-set')
    dataset_labels, dataset = csv_dict_list('datasets/MOV.csv')
    build_pipeline(dataset, dataset_labels, scoring_metric)

    # testing the model on Product Reviews data-set
    print('***********************************************************************************************************')
    print('Testing the model on Product Reviews data-set')
    dataset_labels, dataset = csv_dict_list('datasets/PROD.csv')
    build_pipeline(dataset, dataset_labels, scoring_metric)

    # testing the model on Restaurants Reviews data-set
    print('***********************************************************************************************************')
    print('Testing the model on Restaurants Reviews data-set')
    dataset_labels, dataset = csv_dict_list('datasets/RES.csv')
    build_pipeline(dataset, dataset_labels, scoring_metric)
    print('***********************************************************************************************************')


#all_datasets_results()