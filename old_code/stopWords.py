from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class StopWords:

    def removeStopWords(self, text, lang):

        stop_words = set(stopwords.words(lang))

        word_tokens = word_tokenize(text)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        # filtered_sentence = []
        #
        # for w in word_tokens:
        #     if w not in stop_words:
        #         filtered_sentence.append(w)

        return filtered_sentence


def main():
    stopWords = StopWords()

    entext = "This is a sample sentence, showing off the stop words filtration."
    arText = "حالة الطقسس اليوم في فلسطين ليست جيدة بحسب الارصاد الجوية"

    lang = "arabic"

    filtered_sentence = stopWords.removeStopWords(arText, lang)

    print("Input text: '" + arText + " '")
    print("filtered text: '" + str(filtered_sentence) + " '")


main()
