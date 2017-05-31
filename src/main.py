# coding: utf-8

from os import path
from dataset import Dataset
from lsc_verb_noun_pairs import LSCVerbClasses
from lsc_subj_intransitive_verbs import SubjectIntransitiveVerbClasses
from lsc_subj_obj_transitive_verbs import SubjectObjectTransitiveVerbClasses


"""
This file contains the entire implementation of "Inducing a Semantically Annotated Lexicon via EM-Based Clustering"

The implementation was written by:
[Maurits Bleeker](https://github.com/MBleeker)
[Thijs Scheepers](http://github.com/tscheepers)
[David Zomerdijk](https://github.com/DavidZomerdijk)

Make sure the `data` directory contains the `all_pairs` file before running.
"""


def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    gold_corpus = path.join(data_path, 'gold_deps.txt')
    all_pairs = path.join(data_path, 'all_pairs')

    dataset = Dataset.load(all_pairs, n_test_pairs=3000)

    # Parameter grid
    parameters = [
        # (5, 151),
        # (10, 151),
        # (20, 151),
        (30, 151),
        (40, 151),
        (50, 151),
        (75, 151),
        (100, 151),
        (200, 151),
        (300, 151)
    ]

    # Running the experiment for all parameters in the grid
    for (n_cs, em_iters) in parameters:
        print("------ Clusters: %d ------" % (n_cs))
        print("------ Step 1 ------")
        # Step one also evaluates in between runs
        step1 = LSCVerbClasses(dataset, n_cs=n_cs, em_iters=em_iters, name='all_pairs_lcs')
        step1.train()
        print("------ Step 2 - Intransitive ------")
        step2_1 = SubjectIntransitiveVerbClasses(dataset, step1, em_iters=em_iters, name='all_pairs_intransitive_class')
        step2_1.train()
        # print("------ Step 2 - Transitive ------")
        # step2_2 = SubjectObjectTransitiveVerbClasses(dataset, step1, em_iters=em_iters, name='all_pairs_transitive_class')
        # step2_2.train()


if __name__ == "__main__":
    main()