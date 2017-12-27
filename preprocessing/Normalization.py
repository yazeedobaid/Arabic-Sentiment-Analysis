from pyarabic import araby

class Normalization:

    def normalizeText(self, text):
        normalized_text = araby.strip_tatweel(text)
        normalized_text = araby.strip_tashkeel(normalized_text)
        normalized_text = araby.strip_harakat(normalized_text)
        normalized_text = araby.normalize_hamza(normalized_text)

        return normalized_text