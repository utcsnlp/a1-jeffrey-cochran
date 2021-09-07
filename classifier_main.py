# classifier_main.py

import argparse
import sys
import time
import itertools
from copy import deepcopy
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List, Tuple
from constants import permissible_labels, id_labels
from models import *

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='classifier_main.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args

def map_tag_to_label(tag):
    """
    """
    label = tag.split("-")[-1]
    return label

def transform_for_classification(ner_exs: List[LabeledSentence]):
    """
    :param ner_exs: List of chunk-style NER examples
    :return: A list of NERExamples extracted from the NER data
    """
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        # labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        labels = [map_tag_to_label(tag) for tag in tags]
        yield NERExample([tok.word for tok in labeled_sent.tokens], labels)

def evaluate_classifier(exs: List[NERExample], classifier: NERClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: NERExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            predictions.append(classifier.predict(ex.tokens, idx))
    print_evaluation(golds, predictions)


def compute_metrics(true_positive=None, true_negative=None, false_positive=None, false_negative=None) -> Tuple[float]:
    total = true_negative + true_positive + false_negative + false_positive
    num_pred = true_positive + false_positive
    num_gold = true_positive + false_negative
    #
    acc = float(true_positive + true_negative) / total if total > 0 else 0.0
    prec = float(true_positive) / num_pred if num_pred > 0 else 0.0
    rec = float(true_positive) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    #
    return (acc, prec, rec, f1)

def lookup_label_results(confusion_matrix: dict=None, label: str=None) -> Tuple[int]:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #
    for k in confusion_matrix.keys():
        gold, pred = k
        if pred == label and gold == label:
            TP += confusion_matrix[k]
        elif pred == label and gold != label:
            FP += confusion_matrix[k]
        elif pred != label and gold == label:
            FN += confusion_matrix[k]
        else:
            TN += confusion_matrix[k]
    return (TP, TN, FP, FN)

def report_label_metrics(label: str=None, true_positive: int=None, true_negative: int=None, false_positive: int=None, false_negative: int=None):

    acc, prec, rec, f1 = compute_metrics(
        true_positive=true_positive, 
        true_negative=true_negative, 
        false_positive=false_positive, 
        false_negative=false_negative
    )

    recall_denom = true_positive + false_negative
    precision_denom = true_positive + false_positive
    print_str = """Performance Metrics for %s: Recall: %i / %i = %f, Precision: %i / %i = %f, F1: %f""" % (label, true_positive, recall_denom, rec, true_positive, precision_denom, prec, f1)
    
    print(print_str)

    return (prec, rec, f1)

def report_macro_metrics(accuracy: float=None, recall: float=None, precision: float=None, f1: float=None, num_correct: int=None, num_total: int=None):

    print_str = """Overall Performance Metrics:
----------------
Accuracy: %i / %i = %f
Macro-Recall: %f
Macro-Precision: %f
Macro-F1: %f
""" % (num_correct, num_total, accuracy, recall, precision, f1)
    
    print(print_str)
    
    return

def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    # ----
    confusion_keys = itertools.product(permissible_labels, repeat=2)
    confusion_matrix = {
        k: 0 for k in confusion_keys
    } # NOTE: 
      # The keys that index into the confusion matrix
      # are given as (ground_truth_label, predicted_label) 
    # ----
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    #
    # Populate confusion matrix
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        confusion_matrix[(gold, prediction)] += 1
    #
    # Compute label metrics
    num_named_entity_labels = float(len(permissible_labels)-1)
    num_correct = 0
    rec_macro = 0
    prec_macro = 0
    f1_macro = 0
    for l in permissible_labels:

        TP, TN, FP, FN = lookup_label_results(
            confusion_matrix=confusion_matrix, 
            label=l
        )
        num_correct += TP

        #
        # Only average over named entity categories
        if l != "O":

            prec, rec, f1 = report_label_metrics(
                label=l,
                true_positive=TP, 
                true_negative=TN, 
                false_positive=FP, 
                false_negative=FN
            )
            #
            # accumulate
            prec_macro += prec
            rec_macro += rec
            f1_macro += f1
    #
    # Report macros
    num_total = TP + TN + FP + FN
    acc = num_correct / float(num_total)
    prec_macro /= num_named_entity_labels
    rec_macro /= num_named_entity_labels
    f1_macro /= num_named_entity_labels

    report_macro_metrics(
        accuracy=acc,
        recall=rec_macro,
        precision=prec_macro,
        f1=f1_macro,
        num_correct=num_correct,
        num_total=num_total
    )
    return


def predict_write_output_to_file(exs: List[NERExample], classifier: NERClassifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(prediction) + "\n")
        f.write("\n")
    f.close()


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_classifier(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs)
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train Performance===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev Performance===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))
