# scorer.py

from nerdata import *
import os


# "Blind" a file (used only once to make eng.testb.blind)
def make_blind(file):
    f = open(file)
    o = open(file + ".blind", 'w')
    for line in f:
        if line.strip() == "":
            o.write(line)
        else:
            fields_to_keep = line.split()[:-1]
            o.write(" ".join(fields_to_keep) + " O\n")
    f.close()
    o.close()

# Fixes up the raw CoNLL files that don't have B-... tags to start chunks.
def b_ify(file):
    f = open(file)
    o = open(file + ".withb", 'w')
    last_tag = "O"
    for line in f:
        if line.strip() == "":
            o.write(line)
            last_tag = "O"
        else:
            fields = line.split()
            my_tag = fields[-1]
            my_tag_bio = fields[-1][0]
            my_tag_type = fields[-1][2:] if fields[-1] != "O" else ""
            # If it's an I tag that's not extending the previous thing, should be B
            if my_tag_bio == "I" and my_tag != last_tag:
                o.write(" ".join(fields[:-1]) + " B-" + my_tag_type + "\n")
            else:
                o.write(line)
            last_tag = my_tag
    f.close()
    o.close()

# Reads in test predictions and the test data and scores them
if __name__ == '__main__':

    # make_blind("data/eng.testb")
    # b_ify("data/eng.train-new")
    # b_ify("data/eng.testa-new")
    # b_ify("data/eng.testb-new")
    # exit()

    # Actual scorer code
    student_out_path = "../hw1-student-outputs"
    golds = read_data("data/eng.testb")
    for filename in sorted(os.listdir(student_out_path)):
        if ".out" in filename:
            guesses = read_data(student_out_path + "/" + filename)
            print("Eval on: %s" % filename)
            print_evaluation(golds, guesses)
        else:
            print("Filtering %s" % filename)


