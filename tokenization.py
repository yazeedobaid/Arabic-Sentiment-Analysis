from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenization:

    def setenceLevelTokenize(self, text):
        return sent_tokenize(text)


    def wordLevelTokenize(self, text):
        return word_tokenize(text)



def main():

    tokenizer = Tokenization()

    enText = "Hello Mr. Smith, how are you doing today? The weather is great, " \
           "and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

    print("Tokenizing English text:")
    print("Text is: ' " + enText + " '")
    print("     Sentence level tokenization: " + str(tokenizer.setenceLevelTokenize(enText)))
    print("     Word level tokenization: " + str(tokenizer.wordLevelTokenize(enText)))
    print("---------------------------------------------------------------------------------")

    arText = "مرحبا سيد سميث, كيف حالك اليوم؟ الطقس ممتاز, " \
             "والبايثون رائعة. السماء ممتلئة بالغيوم. لا تأكل الشوكولاتة."

    print("Tokenizing Arabic text:")
    print("Text is: ' " + arText + " '")
    print("     Sentence level tokenization: " + str(tokenizer.setenceLevelTokenize(arText)))
    print("     Word level tokenization: " + str(tokenizer.wordLevelTokenize(arText)))


main()
