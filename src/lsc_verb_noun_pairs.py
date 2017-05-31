# coding: utf-8

from os import path
import numpy as np
import pickle
from eval_pseudo_disambiguation import EvaluationPseudoDisambiguation
from eval_embedding_centroid import EvaluationEmbeddingCentroids


class LSCVerbClasses:
    """
    Latent Semantic Clustering for Verb Classes

    This is the implementation of Step 1
    Verb classes (corresponding to Section 2 in the paper https://arxiv.org/abs/cs/9905008)
    """

    def __init__(self, dataset, n_cs=30, em_iters=50, name='step1'):
        """
        :param dataset: The dataset for which to train
        :param n_cs: Number of classes
        :param em_iters: Iterations
        """
        self.dataset = dataset
        self.n_cs = n_cs
        self.em_iters = em_iters
        self.current_iter = 0
        self.name = name
        self.likelihoods = list()
        self.accuracies = dict()

        self.p_vn = None  # Calculated each EM-Iteration
        self.p_c, self.p_vc, self.p_nc = self.initialize_parameters()

        # Helpers for the dataset
        self.ys_v = [v for (v, n, _, _) in dataset.ys]
        self.ys_n = [n for (v, n, _, _) in dataset.ys]
        self.f_ys = np.array(dataset.f_ys)

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.n_cs, self.em_iters, self.current_iter, self.name, self.likelihoods, \
               self.p_c, self.p_vc, self.p_nc, self.accuracies

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.n_cs, self.em_iters, self.current_iter, self.name, self.likelihoods, \
        self.p_c, self.p_vc, self.p_nc, self.accuracies = state

    def initialize_parameters(self):
        """
        Initialize theta parameters
        :return:
        """
        np.random.seed(1)

        p_c = np.random.rand(self.n_cs)
        p_c /= np.sum(p_c)

        p_vc = np.random.rand(self.dataset.n_vps, self.n_cs)
        p_vc /= np.sum(p_vc, axis=0)

        p_nc = np.random.rand(self.dataset.n_ns, self.n_cs)
        p_nc /= np.sum(p_nc, axis=0)

        return p_c, p_vc, p_nc

    def train(self):
        """
        Train the algorithm
        """

        pseudo_disambiguation = EvaluationPseudoDisambiguation(self.dataset, self, lower_bound=30)
        centroid_evaluator = EvaluationEmbeddingCentroids(self.dataset, self)

        for i in range(self.current_iter, self.em_iters):
            self.current_iter = i

            likelihood = self.em_iter()

            self.likelihoods.append(likelihood)

            if i % 5 == 0 and i != 0:
                acc = pseudo_disambiguation.evaluate()
                self.accuracies[i] = acc
                print('%i: Log-likelihood: %f\tAccuracy:\t%f' % (i, likelihood, acc))

            else:
                print('%i: Log-likelihood: %f' % (i, likelihood))

            if i % 25 == 0:
                centroid_evaluator.evaluate(file_name='%d-%d' % (self.n_cs, i))
                self.store()

    def p_c_vn(self):
        """
        P(c|v, n)
        :return:
        """
        p_c_vn = (self.p_c * self.p_vc[self.ys_v, :] * self.p_nc[self.ys_n, :]).T
        p_vn = np.sum(p_c_vn, axis=0)  # P(v, n)
        p_c_vn /= p_vn  # P(c|v, n)
        return p_c_vn, p_vn

    def em_iter(self):
        """
        Do an EM step
        :return: log-likelihood
        """

        p_c_1 = np.array(self.p_c)
        p_vc_1 = np.array(self.p_vc)
        p_nc_1 = np.array(self.p_nc)

        p_c_vn, p_vn = self.p_c_vn()

        likelihood = np.sum(self.f_ys * np.log(p_vn))

        # d = Sigma_y f(y)p(x|y)
        d = np.sum(self.f_ys * p_c_vn, axis=1)

        for v in range(self.dataset.n_vps):
            # Sigma_y in {v} X N f(y)p(x|y) / d
            ys_per_v = self.dataset.ys_per_vp[v]
            p_vc_1[v, :] = np.sum(self.f_ys[ys_per_v] * p_c_vn[:, ys_per_v], axis=1) / d

        for n in range(self.dataset.n_ns):
            # Sigma_y in N X {v} f(y)p(x|y) / d
            ys_per_n = self.dataset.ys_per_n[n]
            p_nc_1[n, :] = np.sum(self.f_ys[ys_per_n] * p_c_vn[:, ys_per_n], axis=1) / d

        # d / |Y|
        p_c_1 = d / self.dataset.n_ys

        self.p_c = p_c_1
        self.p_vc = p_vc_1
        self.p_nc = p_nc_1

        return likelihood

    def p_n_v(self, n, v):
        """
        p(n|v)
        """

        p_c_vn = self.p_c * self.p_vc[v, :] * self.p_nc
        p_vn = np.sum(p_c_vn.T, axis=0)  # P(v, n) for all n
        p_v = np.sum(p_vn)  # P(v)

        return p_vn[n] / p_v

    def store(self):
        """
        Function to save the class, which we can use for step 2
        :param file_name:
        :return:
        """
        out_path = path.join(
            path.dirname(__file__), '..', 'out',
            '%s-%d-%d.pkl' % (self.name, self.n_cs, self.current_iter)
        )

        pickle.dump(self,  open(out_path, 'wb'))