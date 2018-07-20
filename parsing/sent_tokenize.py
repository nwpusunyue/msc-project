from abc import ABC, abstractmethod
from nltk.tokenize import sent_tokenize
from parsing.genia.sent_tokenize import genia_sent_tokenize


class SentTokenizer(ABC):

    @abstractmethod
    def sent_tokenize(self, text):
        pass


class NLTKSentTokenizer(SentTokenizer):

    def sent_tokenize(self, text):
        return sent_tokenize(text)


class GeniaSentTokenizer(SentTokenizer):

    def sent_tokenize(self, text):
        return genia_sent_tokenize(text)
