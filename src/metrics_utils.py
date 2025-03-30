import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class Similarity:
    def __init__(self, topic_word):
        self.topic_word = topic_word
        self.K = topic_word.shape[0]
        self.W = topic_word.shape[1]

    def get_similarity(self, topic_index):
        word_distribution_i = self.topic_word[topic_index].reshape((1, self.W))
        word_distribution_js = np.delete(self.topic_word, topic_index, axis=0)
        return cosine_similarity(word_distribution_i, word_distribution_js)

    def model_similarity(self, type=None):
        topic_similarity = []
        for k in range(self.K):
            similarity_scores = self.get_similarity(k)
            similarity = np.mean(similarity_scores)
            if type == "max":
                similarity = np.max(similarity_scores)
            topic_similarity.append(similarity)
        return topic_similarity


class Coherence:

    def __init__(self, counts_vector):
        self.counts_vector = counts_vector
        self.get_marginals()

    def get_marginals(self):
        X = self.counts_vector.copy()
        X[X > 0] = 1
        D = X.shape[0]
        nw = X.sum(axis=0)
        self.pw = nw / D

        coo = np.transpose(X).dot(X)
        self.pcoo = coo / D

    def get_npmi(self, ix, jx):
        coo = self.pcoo[ix, jx]
        if coo > 0:
            pmi = np.log(coo / (self.pw[0, ix] * self.pw[0, jx]))
            npmi = pmi / -np.log(coo)
            return float(npmi)
        else:
            return -1

    def get_topic_coherence(self, word_indexes):

        coherence_scores = []
        for i in range(len(word_indexes) - 1):
            for j in range(i + 1, len(word_indexes)):
                ix = word_indexes[i]
                jx = word_indexes[j]
                coherence_scores.append(self.get_npmi(ix, jx))
        return float(np.mean(coherence_scores))

    def model_coherence(self, list_word_indexes):
        topic_coherence = []
        for word_indexes in list_word_indexes:
            topic_coherence.append(self.get_topic_coherence(word_indexes))
        return topic_coherence

