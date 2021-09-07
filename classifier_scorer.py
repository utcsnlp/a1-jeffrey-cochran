# classifier_scorer.py

from classifier_main import *
import os


# Reads in test predictions and the test data and scores them
if __name__ == '__main__':
    submissions_dir = "mini1outputs/"
    gold_exs = list(transform_for_classification(read_data("data/eng.testb")))
    golds = [label for ex in gold_exs for label in ex.labels]
    for filename in os.listdir(submissions_dir):
        if ".out" in filename:
            print("Eval on: %s" % filename)
            with open(submissions_dir + "/" + filename) as f:
                student_outputs = f.readlines()
            curr_ex = 0
            guesses = []
            for line in student_outputs:
                line = line.strip()
                if len(line) > 0:
                    # print(line)
                    guesses.append(int(line[-1]))
            # print(golds)
            # print(guesses)
            print_evaluation(golds, guesses)
        else:
            print("Filtering %s" % filename)


