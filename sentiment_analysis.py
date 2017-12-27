from sklearn.linear_model import SGDClassifier
from utilities.model_persistence import *

class sentiment_analysis:
    # -----------------------------------------------------------------------------------------------------------------------
    # Training a classifier on the data-set.
    #
    #   PARAMETERS:
    #       - X_train_tfidf: the tf-idf list of the terms in data-set
    #       - dataset_labels: the labels of the data-set examples
    #       - classifierType: the classifier type, algorithm
    #       - modelFilePath: the path to the classifier to serialize after finishing training
    #
    #   RETURNS:
    #       - classifier: the trained classifier object
    # -----------------------------------------------------------------------------------------------------------------------
    def sentiment_analysis_train(self,
                                 X_train_tfidf=None,
                                 dataset_labels=None,
                                 classifierType='svm',
                                 modelFilePath='models/svm.pickle'):
        # Check if the data-set and its labels are provided
        if X_train_tfidf == None or dataset_labels == None:
            print('No data-set or data-set labels are provided!')
            exit(1)

        # An SVM classifier to use for the learning process
        classifier = SGDClassifier(loss='hinge', penalty='l2',
                                   alpha=1e-3, random_state=42,
                                   max_iter=5, tol=None)

        # Fitting the classifier using the data-set
        classifier.fit(X_train_tfidf, dataset_labels)

        # Serializing the classifier
        serializeClassifier(classifier, modelFilePath)

        return classifier


    # -----------------------------------------------------------------------------------------------------------------------
    # Train a classifier on new text data. The method prints the result of the classification
    #
    #   PARAMETERS:
    #       - input_text: the testing input text to classify
    #       - classifier: the classifier object
    #       - count_vect: count vector of the terms in data-set
    #       - X_train_tfidf: the tf-idf list of the terms in data-set
    #       - tfidf_transformer: the tf-idf transformer object
    #       - categories: categories of the target class labels
    #
    #   RETURNS:
    #       - No return value!
    # -----------------------------------------------------------------------------------------------------------------------
    def sentiment_analysis_test(self,
                                input_text,
                                classifier,
                                count_vect,
                                X_train_tfidf,
                                tfidf_transformer,
                                categories):
        # Check if the data-set and its labels are provided
        if input_text == None or classifier == None:
            print('No testing text or classifier provided!')
            exit(1)

        X_new_counts = count_vect.transform(input_text)

        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        predicted = classifier.predict(X_new_tfidf)

        for doc, category in zip(input_text, predicted):
            if category == 1 or category == -1:
                print('%r => %s' % (doc, categories['Positive' if category else 'Negative']))
            else:
                print('%r => %s' % (doc, categories[category]))
