import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


class SentimentAnalysis:

    def loadDataSet(self, documents):
        random.shuffle(documents)

        all_words = []
        for w in movie_reviews.words():
            all_words.append(w.lower())

        all_words = nltk.FreqDist(all_words)

        return all_words

    def find_features(self, document, word_features):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    def serializeClassifier(self, classifier, modelFilePath):
        save_classifier = open(modelFilePath, "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

    def loadClassifier(self, modelFilePath):
        classifier_f = open(modelFilePath, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()

        return classifier


    def trainClassifier(self, trainingData, classifierType):

        classifier = None

        if classifierType == "NaiveBayes":
            classifier = nltk.NaiveBayesClassifier.train(trainingData)
        elif classifierType == "MultinomialNB":
            classifier = SklearnClassifier(MultinomialNB())
            classifier.train(trainingData)
        elif classifierType == "BernoulliNB":
            classifier = SklearnClassifier(BernoulliNB())
            classifier.train(trainingData)
        elif classifierType == "LogisticRegression":
            classifier = SklearnClassifier(LogisticRegression())
            classifier.train(trainingData)
        elif classifierType == "SGDClassifier":
            classifier = SklearnClassifier(SGDClassifier())
            classifier.train(trainingData)
        elif classifierType == "SVC":
            classifier = SklearnClassifier(SVC(probability=True))
            classifier.train(trainingData)
        elif classifierType == "LinearSVC":
            classifier = SklearnClassifier(LinearSVC())
            classifier.train(trainingData)
        elif classifierType == "NuSVC":
            classifier = SklearnClassifier(NuSVC(probability=True))
            classifier.train(trainingData)

        return classifier


if __name__ == "__main__":
    sentimentAnalysis = SentimentAnalysis()

    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    all_words = sentimentAnalysis.loadDataSet(documents)

    word_features = list(all_words.keys())[:3000]

    featuresets = [(sentimentAnalysis.find_features(rev, word_features), category) for (rev, category) in documents]

    # set that we'll train our classifier with
    trainingSet = featuresets[:1900]
    # set that we'll test against.
    testingSet = featuresets[1900:]

    classifierType = "SVC"
    modelFilePath = "models/" + classifierType + ".pickle"

    classifier = sentimentAnalysis.trainClassifier(trainingSet, classifierType)
    sentimentAnalysis.serializeClassifier(classifier, modelFilePath)

    classifier = sentimentAnalysis.loadClassifier(modelFilePath)

    print("Classifier '" + classifierType + "' accuracy percent:", (nltk.classify.accuracy(classifier, testingSet)) * 100)

    print("classes lables: " + str(classifier.labels()))


    # to classify new sentence use

    posReview = "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"
    print(posReview)
    testFeatures = sentimentAnalysis.find_features(posReview, word_features)
    print("posReview result: " + classifier.classify(testFeatures))
    labelsProbabiliies = classifier.prob_classify(testFeatures)
    for label in labelsProbabiliies.samples():
        print("%s: %f" % (label, labelsProbabiliies.prob(label)))

    print("------------------------------------------------------------------------------------------------------------")

    negReview = "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. \n" \
                "Horrible movie, 0/10"
    print(negReview)
    testFeatures = sentimentAnalysis.find_features(negReview, word_features)
    print("negReview result: " + classifier.classify(testFeatures))
    labelsProbabiliies = classifier.prob_classify(testFeatures)
    for label in labelsProbabiliies.samples():
        print("%s: %f" % (label, labelsProbabiliies.prob(label)))

