from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize

class Stemming:
    def __init__(self):
        self.st = ISRIStemmer()

    def stemWord(self, text):
        word_tokens = word_tokenize(text)
        filtered_sentence = [self.st.stem(w) + ' ' for w in word_tokens]

        return ''.join(filtered_sentence)

