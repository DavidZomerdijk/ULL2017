# coding: utf-8

from os import path
import numpy as np
import pickle


class SubjectObjectTransitiveVerbClasses:
    """
    Clustering for subjects and objects for transitive verbs
    This is the implementation of Step 2
    Verb classes (corresponding to Section 4.2 in the paper https://arxiv.org/abs/cs/9905008)
    """

    def __init__(self, dataset, model, em_iters=50, name='step2-2'):
        """
        :param dataset: The dataset for which to train
        :param n_cs: Number of classes
        :param em_iters: Iterations
        """
        self.dataset = dataset
        self.model = model
        self.em_iters = em_iters
        self.name = name
        self.current_iter = 0
        self.likelihoods = list()

        self.p_c = self.initialize_parameters()

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.em_iters, self.current_iter, self.name, self.likelihoods, self.p_c

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.em_iters, self.current_iter, self.name, self.likelihoods, self.p_c = state

    def initialize_parameters(self):
        """
        Initialize theta parameters
        :return:
        """
        np.random.seed(1)

        p_c = np.random.rand(self.model.n_cs, self.model.n_cs)
        p_c /= np.sum(p_c)

        return p_c

    def train(self):
        """
        Train the algorithm
        """

        fs = np.array(self.dataset.f_ws)
        ws_s = [s for (s, o) in self.dataset.ws]
        ws_o = [o for (s, o) in self.dataset.ws]

        for i in range(self.em_iters):
            self.current_iter = i

            likelihood = self.em_iter(fs, ws_s, ws_o)
            self.likelihoods.append(likelihood)

            print('%i: Log-likelihood: %f' % (i, likelihood))

            if i % 10 == 0:
                self.store()

    def em_iter(self, fs, ws_s, ws_o):
        """
        Do an EM step
        :return: log-likelihood
        """

        p_nc1 = np.expand_dims(self.model.p_nc[ws_s, :], axis=1)
        p_nc2 = np.expand_dims(self.model.p_nc[ws_o, :], axis=2)

        p_c_n = (self.p_c * (p_nc1 * p_nc2)).T # p(c)P_LC(n1|c1)P_LC(n2|c2)
        p_n = np.sum(p_c_n, axis=(0, 1)) # P(n1,n2)
        p_c_n /= p_n  # P(c1,c2|n1,n2)

        likelihood = np.sum(fs * np.log(p_n))

        self.p_c = np.sum(fs * p_c_n, axis=2) / np.sum(fs)

        return likelihood

    def store(self):
        """
        Function to save the class
        :param file_name:
        :return:
        """
        out_path = path.join(
            path.dirname(__file__), '..', 'out',
            '%s-%d-%d.pkl' % (self.name, self.model.n_cs, self.current_iter)
        )

        pickle.dump(self, open(out_path, 'wb'))