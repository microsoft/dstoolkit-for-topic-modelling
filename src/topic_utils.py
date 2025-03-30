import numpy as np
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt

class TopicAnalyser:

    def __init__(self, topic_mixtures, topics, vocab , ntop=10, dtop=10):
        self.topics = topics
        self.topic_mixtures = topic_mixtures
        self.vocab = vocab
        self.ntop = ntop
        self.dtop = dtop
        self.V = len(vocab)
        self.K = topics.shape[0]
        self.W = topics.shape[1]
        self.D = topic_mixtures.shape[0]
        self.all_docs_top_topic = np.argmax(self.topic_mixtures, 1)
        self.all_docs_top_prob = np.max(self.topic_mixtures, 1)
    
    def get_topic(self, topic_index):
        return self.topics[topic_index] 
    
    def get_document(self, doc_index):
        return self.topic_mixtures[doc_index] 

    def get_top_word_index(self, topic_index):
        word_distribution= self.get_topic(topic_index)
        return np.argsort(word_distribution)[::-1][:self.ntop]

    def get_top_word_prob(self, topic_index):
        word_distribution= self.get_topic(topic_index)
        return np.sort(word_distribution)[::-1][:self.ntop]
    
    def get_top_words(self, topic_index):
        topic_word_index= self.get_top_word_index(topic_index)
        return [self.vocab[ix] for ix in topic_word_index]
    
    def get_all_top_word_index(self):
        all_top_word_index = []
        for tx in range(self.K):
            all_top_word_index.append(self.get_top_word_index(tx))
        return all_top_word_index
    
    def get_top_doc_index(self, topic_index, dtop = None):
        topic_doc_indexes = np.where(self.all_docs_top_topic == topic_index)[0]
        topic_doc_probs = self.all_docs_top_prob[topic_doc_indexes]
        if dtop is None:
            dtop= self.dtop
        topic_top_doc_indexes = topic_doc_indexes[np.argsort(topic_doc_probs)][:dtop]
        return [int(ix) for ix in topic_top_doc_indexes]
    
    def plot_topic_distribution(self, k):
        prob = self.get_topic(k)
        word_ids = np.arange(self.V)
        step = np.arange(self.V, round(self.V/10))
        plt.xticks(step, step)
        plt.bar(word_ids, prob, width=1)
        plt.ylabel('p(w)')
        plt.xlabel('vocabulary')

    def plot_document_distribution(self, d):
        prob = self.get_document(d)
        topic_ids = range(self.K)
        plt.bar(topic_ids, prob, width=1)
        plt.xticks(topic_ids, topic_ids)
        plt.ylabel('p(t)')
        plt.xlabel('topic index')