# coding: utf-8

from os import path
from dataset import Dataset
from lsc_verb_noun_pairs import LSCVerbClasses
from lsc_subj_intransitive_verbs import SubjectIntransitiveVerbClasses
from lsc_subj_obj_transitive_verbs import SubjectObjectTransitiveVerbClasses
from vae_verb_noun_pairs import VAEVerbClasses


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

    # dataset = Dataset.load(gold_corpus, n_test_pairs=500)
    dataset = Dataset.load(all_pairs, n_test_pairs=10000)

    # Parameter grid
    parameters = [
        (5, 100),
        (10, 100),
        (20, 100),
        (30, 100),
        (35, 100),
        (40, 100),
        (50, 100),
        (75, 100),
        (100, 100),
        (200, 100),
        (300, 100),
        (400, 100),
        (500, 100),
    ]

    # Running the experiment for all parameters in the grid
    for (n_cs, em_iters) in parameters:
        print("------ Clusters: %d ------" % (n_cs))
        print("------ Step 1 - LSC ------")
        # Step one also evaluates in between runs
        step1 = LSCVerbClasses(dataset, n_cs=n_cs, em_iters=em_iters + 1, name='all_pairs_lcs')
        step1.train()
        print("------ Step 2 - Lexical Induction - Intransitive ------")
        step2_1 = SubjectIntransitiveVerbClasses(dataset, step1, em_iters=em_iters + 1, name='all_pairs_intransitive_class')
        step2_1.train()
        # print("------ Step 2 - Lexical Induction - Transitive ------")
        # step2_2 = SubjectObjectTransitiveVerbClasses(dataset, step1, em_iters=em_iters + 1, name='all_pairs_transitive_class')
        # step2_2.train()

    print("------ Training the VAE ------")
    model = VAEVerbClasses(dataset)
    model.train()
    # model = VAEVerbClasses(dataset, hidden_dim=50, latent_dim=10)
    # model.train(batch_size=32, n_epochs=10000, lower_bound_pde=3)


if __name__ == "__main__":
    main()