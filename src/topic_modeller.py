import numpy as np
import json
from data_utils import TextPreProcessor, CorpusProcessor
from lda import LDA
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import argparse

def parse_config_file(file_path):
    configurations = []
    with open(file_path, 'r') as file:
        for line in file:
            config = json.loads(line)
            configurations.append(config)
    return configurations

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", 
        help="Path to JSONL configuration file", 
        required=True
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    configurations = parse_config_file(args.config)

    for idx, cfg in enumerate(configurations, start=1):
        dataset_name = cfg["dataset_name"]
        alpha = cfg["alpha"]
        eta = cfg["eta"]
        n_topics = cfg["n_topics"]
        n_iters = cfg["n_iters"]
        run_id = cfg["run_id"]

        dataset = load_dataset(f"{dataset_name}")
        documents=[]
        for d in dataset['train']['overview']:
            if d is not None :
                documents.append(d)

        # prepreprocess
        tp = TextPreProcessor()
        documents = tp.preprocess(documents)

        # process documents
        cp = CorpusProcessor(max_relative_frequency = 0.9, min_absolute_frequency =  5)
        cp.process(documents)

        # get and save the vocab
        vocab = cp.get_vocab()
        file = open('./data/{name}'.format(name='vocab'), 'wb')
        pickle.dump(vocab, file)
        file.close()

        X = cp.get_vectorised_documents()
        dataset = dataset_name.replace('/', '_')
        name = f'model_{run_id}.pkl'
        topic_model = LDA(n_topics, n_iters, alpha, eta)
        topic_model.fit(X)

        file = open('./models/{name}'.format(name=name), 'wb')
        pickle.dump(topic_model, file)
        file.close()
