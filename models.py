# models.py

import time
import numpy as np
from utils import *
from collections import Counter
from nerdata import *
from optimizers import *
from typing import List
from constants import id_labels, label_idx, label_vectors

class CountBasedClassifier(object):
    """
    Classifier that takes counts of how often a word was observed with different labels.
    Unknown tokens or ties default to O label, and then person, location, organization and then MISC.
    Attributes:
        per_counts: how often each token occurred with the label PERSON in training
        loc_counts: how often each token occurred with the label LOC in training
        org_counts: how often each token occurred with the label ORG in training
        misc_counts: how often each token occurred with the label MISC in training
        null_counts: how often each token occurred with the label O in training
    """
    def __init__(self, 
            per_counts: Counter, 
            loc_counts: Counter,
            org_counts: Counter,
            misc_counts: Counter,
            null_counts: Counter
            ):
        self.per_counts = per_counts
        self.loc_counts = loc_counts
        self.org_counts = org_counts
        self.misc_counts = misc_counts
        self.null_counts = null_counts

    def predict(self, tokens: List[str], idx: int):
        per_count = self.per_counts[tokens[idx]]
        loc_count = self.loc_counts[tokens[idx]]
        org_count = self.org_counts[tokens[idx]]
        misc_count = self.misc_counts[tokens[idx]]
        null_count = self.null_counts[tokens[idx]]
        max_count = max(per_count, loc_count, org_count, misc_count, null_count)
        if null_count == max_count:
            return 'O'
        elif per_count == max_count:
            return 'PER'
        elif loc_count == max_count:
            return 'LOC'
        elif org_count == max_count:
            return 'ORG'
        elif misc_count == max_count:
            return 'MISC'
        else:
            print('ERROR?')    
        return 'O'

def train_count_based_classifier(ner_exs: List[NERExample]) -> CountBasedClassifier:
    """
    :param ner_exs: training examples to build the count-based classifier from
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedClassifier using counts collected from the given examples
    """
    per_counts = Counter()
    loc_counts = Counter()
    org_counts = Counter()
    misc_counts = Counter()
    null_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 'PER':
                per_counts[ex.tokens[idx]] += 1.0
            elif ex.labels[idx] == 'LOC':
                loc_counts[ex.tokens[idx]] += 1.0
            elif ex.labels[idx] == 'ORG':
                org_counts[ex.tokens[idx]] += 1.0
            elif ex.labels[idx] == 'MISC':
                misc_counts[ex.tokens[idx]] += 1.0
            else:
                null_counts[ex.tokens[idx]] += 1.0
    return CountBasedClassifier(per_counts, loc_counts, org_counts, misc_counts, null_counts)



class NERClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given NERExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        raise Exception("Implement me!")

def train_classifier(ner_exs: List[NERExample]) -> NERClassifier:
    raise Exception("Implement me!")
