# coding: utf-8

import random


class EvaluationPseudoDisambiguation:
    """
    Evaluation for Pseudo-Disambiguation (section 3.1)
    """

    def __init__(self, dataset, model, lower_bound=30, upper_bound=3000):

        self.dataset = dataset
        self.model = model
        self.zs = list()  # tripples

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.build_tripples()

    def build_tripples(self):

        random.seed(1)

        for i, (v, n) in enumerate(self.dataset.ys_test):

            # don't consider if upper or lower bound limits are not satisfied
            if self.dataset.f_n[n] < self.lower_bound or self.dataset.f_n[n] > self.upper_bound:
                continue

            if self.dataset.f_vp[v] < self.lower_bound or self.dataset.f_vp[v] > self.upper_bound:
                continue

            # select v' until one found that satisfies requirements
            v_accent = None
            while v_accent is None:
                v_considered = random.choice(range(len(self.dataset.vs)))

                yp = (v_considered, n)

                # don't consider if upper or lower bound limits are not satisfied
                if self.dataset.f_v[v_considered] < self.lower_bound or \
                   self.dataset.f_v[v_considered] > self.upper_bound:
                    continue

                # can't be in the train or test set icm with n
                if yp in self.dataset.ys_dict or yp in self.dataset.ys_test_dict:
                    continue

                v_accent = v_considered

            z = (v, n, v_accent)
            self.zs.append(z)

        print("\tTripples created (test):\t%d" % len(self.zs))

    def evaluate(self):

        if len(self.zs) == 0:
            return 0.0

        success = 0.0

        for (v, n, v_accent) in self.zs:
            if self.model.p_n_v(n, v) > self.model.p_n_v(n, v_accent):
                success += 1.

        return success / float(len(self.zs))