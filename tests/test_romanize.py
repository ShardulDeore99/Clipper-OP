import unittest

from streamclipper.romanize import casual_variants, has_devanagari, romanize


class RomanizeTests(unittest.TestCase):
    def test_common_hinglish_words(self):
        cases = {"मैं": "mai", "नहीं": "nahi", "क्या": "kya", "बहुत": "bahut", "समझना": "samajhna",
                 "है": "hai", "हैं": "hai", "और": "aur", "तुम": "tum", "मज़ा": "maza", "कहानी": "kahani",
                 "चलता": "chalta", "इंसान": "insan", "ज्ञान": "gyan", "आज": "aaj", "सुबह": "subah",
                 "मतलब": "matlab", "सबसे": "sabse", "कमरा": "kamra", "ज़िंदगी": "zindagi",
                 "लड़की": "ladki", "आऊँ": "aau", "क्यों": "kyu", "में": "mein", "कहाँ": "kaha",
                 "मदद": "madad", "कमल": "kamal", "संभव": "sambhav"}
        for dev, roman in cases.items():
            self.assertEqual(romanize(dev), roman, dev)

    def test_the_casual_example_sentence(self):
        self.assertEqual(romanize("मैं क्यों आऊँ नहीं तू जा"), "mai kyu aau nahi tu ja")

    def test_roman_variants_are_folded_to_the_same_style(self):
        self.assertEqual(casual_variants("Main kyun aaun, nahin hain"), "Main kyu aau, nahi hai")
        self.assertEqual(casual_variants("Hello everyone"), "Hello everyone")

    def test_loanwords_and_mixed_text(self):
        self.assertEqual(romanize("गेम"), "game")
        self.assertEqual(romanize("ये game बहुत मस्त है।"), "ye game bahut mast hai.")

    def test_english_and_punctuation_untouched(self):
        self.assertEqual(romanize("Let's go, bro!"), "Let's go, bro!")
        self.assertFalse(has_devanagari("Let's go"))
        self.assertTrue(has_devanagari("है"))


if __name__ == "__main__":
    unittest.main()
