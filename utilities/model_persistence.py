import pickle

#-----------------------------------------------------------------------------------------------------------------------
# Serialize a classifier to hard-disk.
#
#   PARAMETERS:
#       - classifier: classifier object to serialize
#       - modelFilePath: path in hard-disk to serialize classifier to
#
#   RETURNS:
#       - No return value!
#-----------------------------------------------------------------------------------------------------------------------
def serializeClassifier(classifier, modelFilePath):
    save_classifier = open(modelFilePath, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


#-----------------------------------------------------------------------------------------------------------------------
# Read a serialized classifier from hard-disk
#
#   PARAMETERS:
#       - modelFilePath: path to the serialized model in hard-disk
#
#   RETURNS:
#       - classifier: the classifier object
#-----------------------------------------------------------------------------------------------------------------------
def readSerializedClassifier(modelFilePath):
    save_classifier = open(modelFilePath, "rb")
    classifier = pickle.load(save_classifier)
    return classifier