import os
import re
from typing import List

import numpy as np
from scipy.sparse import csr_matrix


class TextPreProcessor:
    """
    A class used to preprocess text data

    Methods
    -------
    preprocess(corpus:List[str])
        Preprocesses each text in the whole corpus in the following way: lowercase text, removes all non-alphabetic characters, converts to list of words.
    """

    def _preprocess(self, text: str) -> List:
        """Preprocesses document the following way: lowercase text, removes all non-alphabetic characters, converts to list of words.

        Parameters
        ----------
        text : str
            The text to preprocess

        Returns
        -------
        list
            a list of words(strings)
        """
        text = text.lower()
        text = re.sub(r"[^a-z]", " ", text.strip())
        text = re.sub("\s\s+", " ", text)
        text = text.split()
        return text

    def preprocess(self, corpus: List[str]) -> List:
        """Wrapper for _preprocess method, takes the whole corpus as an argument and calls _preprocess for each document in corpus.

        Parameters
        ----------
        corpus : List[str]
            The corpus to preprocess: list of strings

        Returns
        -------
        list
            preprocessed corpus: a list of preprocessed texts(strings)
        """
        self.processed_corpus = []
        for doc in corpus:
            self.processed_corpus.append(self._preprocess(doc))
        return self.processed_corpus


class CorpusProcessor:
    """
    A class used to preprocess the corpus

    Attributes
    ----------
    max_relative_frequency : float
        maximum relative frequency (e.g. 0.8 = 80% of corpus words is this word) ith which the word occur in the corpus to not to be excluded from vocabulary
    min_absolute_frequency : int
        minimum absolute frequency (e.g. 10 = the word should occur at least 10 times to be in vocabulary)
        with which the word occur in the corpus to not to be excluded from vocabulary
    stop_words : list
        the list of stopwords, which will be excluded from the corpus

    Methods
    -------
    process(corpus:List[str])
        Processes the corpus:
            - get all unique words and their counts
            - filter words - exclude stopwords, too frequent and too rare words
            - create vocabulary in the format {i: word}
            - process documents: convert every doc in list of words from vocabulary and log empty docs if there are some
            - convert corpus to sparse matrix representation (vectorization): row = doc_id, col = word_id, value = count of the word in the doc.

    get_vectorised_documents()
        Returns the vectorised corpus (sparse matrix representation).

    get_vocab()
        Returns vocabulary of the corpus.

    get_documents()
        Returns filtered documents of the corpus.
    """

    def __init__(
        self,
        max_relative_frequency: float = 0.0,
        min_absolute_frequency: int = 0,
        stop_words: List = None,
    ):
        """
        Parameters
        ----------
        max_relative_frequency : float
            maximum relative frequency (e.g. 0.8 = 80% of corpus words is this word) ith which the word occur in the corpus to not to be excluded from vocabulary
        min_absolute_frequency : int
            minimum absolute frequency (e.g. 10 = the word should occur at least 10 times to be in vocabulary)
            with which the word occur in the corpus to not to be excluded from vocabulary
        stop_words : list
            the list of stopwords, which will be excluded from the corpus
        """
        self.max_relative_frequency = max_relative_frequency
        self.min_absolute_frequency = min_absolute_frequency
        if stop_words:
            self.stop_words = stop_words
        else:
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, "data/stop_words_en.txt")
            stop_words_file = open(path, "r")
            self.stop_words = stop_words_file.read().split("\n")

    def process(self, corpus: List[str]):
        """Processes the corpus.

        Parameters
        ----------
        corpus : List[str]
            The corpus to preprocess: list of strings

        Attributes
        ----------
        D : int
            length of the corpus (amount of docs in it)
        corpus : List[str]
            corpus
        """
        self.D = len(corpus)
        self.corpus = corpus
        self._words_in_corpus()
        self._filter_words()
        self._create_vocab()
        self._build_documents()
        self._vectorize()

    def _words_in_corpus(self):
        """Counts words in corpus.

        Attributes:
        ----------
        words_in_corpus : dict
            dictionary of words and their frequency in corpus
        """
        # n: number of times a word appears, d: number of documents a word appears in
        self.words_in_corpus = {}
        for doc in self.corpus:
            words = np.unique(doc)
            for word in words:
                word = str(word)
                if word in self.words_in_corpus.keys():
                    self.words_in_corpus[word] += 1
                else:
                    self.words_in_corpus[word] = 1

    def _filter_words(self):
        """Filters words in corpus.

        Attributes:
        ----------
        excluded_words : dict
            dictionary of excluded words and their frequency in corpus
        """
        self.excluded_words = {}
        for key in self.words_in_corpus.keys():
            if self._is_out_of_vocab(key, self.words_in_corpus[key]):
                self.excluded_words[key] = self.words_in_corpus[key]
        for key in self.excluded_words.keys():
            self.words_in_corpus.pop(key)

    def _is_out_of_frequency(self, document_count):
        """Checks if word is too rare or too frequent."""
        high_frequency = document_count / self.D >= self.max_relative_frequency
        low_frequency = document_count <= self.min_absolute_frequency
        return high_frequency or low_frequency

    def _is_stop_word(self, word: str):
        """Checks if word is in stop words list"""
        return word in self.stop_words

    def _is_out_of_vocab(self, word: str, document_count: int):
        """Checks if word is either stopword or out of frequency"""
        return self._is_stop_word(word) or self._is_out_of_frequency(document_count)

    def _create_vocab(self):
        """Creates vocabulary.

        Attributes:
        ----------
        words_in_vocab : list
            list of words in vocabulary
        vocab : dict
            vocabulary - dictionary in the format of {index:word}
        W : int
            vocabulary size
        """
        self.words_in_vocab = list(self.words_in_corpus.keys())
        self.vocab = {i: word for i, word in enumerate(self.words_in_vocab)}
        self.W = len(self.vocab)

    def _refine_document(self, doc: List[str]):
        """Filters the documents to keep only vocabulary words in it."""
        return [word for word in doc if word in self.words_in_corpus]

    def _build_documents(self):
        """Converts documents to lists of in-vocabulary words.

        Attributes:
        ----------
        documants : list
            list of processed documents
        D : int
            processed corpus size (number of docs)
        """
        self.documents = []
        log_doc_with_no_words_in_vocab = []
        for ix, doc in enumerate(self.corpus):
            words = self._refine_document(doc)
            if len(words) > 0:
                self.documents.append(words)
            else:
                log_doc_with_no_words_in_vocab.append(ix)
        self.D = len(self.documents)

    def _get_word_id(self, word: str):
        """Returns id of a word in vocabulary."""
        return self.words_in_vocab.index(word)

    def _vectorize(self):
        """Corpus vectorization - converts corpus to sparse matrix.

        Attributes:
        ----------
        vectorised_documents : csr_matrix
            vectorized representation of corpus.
        """
        row = []
        col = []
        data = []
        for ix, doc in enumerate(self.documents):
            words, counts = np.unique(doc, return_counts=True)
            word_ids = [self._get_word_id(word) for word in words]
            row.extend(np.repeat(ix, len(words)))
            col.extend(word_ids)
            data.extend(counts)
        self.vectorised_documents = csr_matrix(
            (data, (row, col)), shape=(self.D, self.W)
        )

    def get_vectorised_documents(self):
        """Returns the vectorised corpus (sparse matrix representation)."""
        return self.vectorised_documents

    def get_vocab(self):
        """Returns vocabulary of the corpus."""
        return self.vocab

    def get_documents(self):
        """Returns filtered documents of the corpus."""
        return self.documents
