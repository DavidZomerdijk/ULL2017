# coding: utf-8
from collections import defaultdict
from os import path
from sys import stdout
import numpy as np
import pickle


class Dataset:
    """
    This class contains all needed information from the dataset
    neatly structured
    """

    @classmethod
    def load(cls, file_path, max_lines=None):
        """
        Loads .pkl file if available, otherwise creates dataset
        :return: Dataset
        """

        pickle_path = cls.pickle_path(file_path, max_lines=max_lines)

        if path.isfile(pickle_path):
            print("Loading dataset (%s) from disk" % pickle_path)
            dataset = pickle.load(open(pickle_path, 'rb'))
        else:
            dataset = Dataset(file_path, max_lines=max_lines)

        print("Dataset ready")
        print("\tUnique verbs:\t%d" % dataset.n_vs)
        print("\tUnique nouns:\t%d" % dataset.n_ns)
        print("\tUnique pairs:\t%d" % dataset.n_ys)

        return dataset

    @classmethod
    def pickle_path(cls, file_path, max_lines=None):
        """
        Returns the path of the pickle file
        """

        if max_lines is None:
            return file_path + '.pkl'
        else:
            return file_path + '-%d.pkl' % max_lines

    def __init__(self, file_path, max_lines=None):
        """
        Initialize a Dataset and read it in
        """

        lines = self.read_lines(file_path)

        # Token-index lookup
        self.ns = list()
        self.vs = list()
        self.ys = list()  # v,n pairs
        self.f_ys = list()  # frequencies

        if max_lines is None or max_lines > len(lines):
            n_lines = len(lines)
        else:
            n_lines = max_lines

        for i, ln in enumerate(lines):

            if i > n_lines:
                break

            if i % 1000 == 0:
                stdout.write("\rReading corpus… %6.2f%%" % ((100 * i) / float(n_lines),))
                stdout.flush()

            if len(ln) == 2:
                vt, nt = ln

                # -------------------------
                # Datastructures for step 1
                # -------------------------

                if nt not in self.ns:
                    n = len(self.ns)
                    self.ns.append(nt)
                else:
                    n = self.ns.index(nt)

                if vt not in self.vs:
                    v = len(self.vs)
                    self.vs.append(vt)
                else:
                    v = self.vs.index(vt)

                yp = (v, n)

                if yp not in self.ys:
                    y = len(self.ys)
                    self.ys.append(yp)
                    self.f_ys.append(1)
                else:
                    y = self.ys.index(yp)
                    self.f_ys[y] += 1

        self.ys_per_v = defaultdict(list)
        self.ys_per_n = defaultdict(list)

        for y, (v, n) in enumerate(self.ys):
            if y not in self.ys_per_v[v]:
                self.ys_per_v[v].append(y)
            if y not in self.ys_per_n[n]:
                self.ys_per_n[n].append(y)

        # Lengths
        self.n_vs = len(self.vs)
        self.n_ns = len(self.ns)
        self.n_ys = len(self.ys)

        print("\rDataset read")

        self.store(file_path, max_lines)

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.ns, self.vs, self.f_ys, self.ys, self.ys_per_v, self.ys_per_n, self.n_vs, self.n_ns, self.n_ys

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.ns, self.vs, self.f_ys, self.ys, self.ys_per_v, self.ys_per_n, self.n_vs, self.n_ns, self.n_ys = state

    def store(self, file_path, max_lines):
        """
        Stores dataset to .pkl file
        """
        pickle.dump(self, open(Dataset.pickle_path(file_path, max_lines=max_lines), 'wb'))

    def read_lines(self, file_path):
        """
        Read a file as a list of tuples
        """
        with open(file_path, 'r') as f:
            return [tuple(ln.strip().split()) for ln in f]


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

        self.p_vn = None  # Calculated each EM-Iteration
        self.p_c, self.p_vc, self.p_nc = self.initialize_parameters()

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.n_cs, self.em_iters, self.current_iter, self.name, self.likelihoods, \
               self.p_c, self.p_vc, self.p_nc

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.n_cs, self.em_iters, self.current_iter, self.name, self.likelihoods, \
        self.p_c, self.p_vc, self.p_nc = state

    def initialize_parameters(self):
        """
        Initialize theta parameters
        :return:
        """
        np.random.seed(1)

        p_c = np.random.rand(self.n_cs)
        p_c /= np.sum(p_c)

        p_vc = np.random.rand(self.dataset.n_vs, self.n_cs)
        p_vc /= np.sum(p_vc, axis=0)

        p_nc = np.random.rand(self.dataset.n_ns, self.n_cs)
        p_nc /= np.sum(p_nc, axis=0)

        return p_c, p_vc, p_nc

    def p_c_vn(self, c, v, n):
        """
        p(c|v, n)
        """

        return self.p_c[c] * self.p_vc[v, c] * self.p_nc[n, c] / self.p_vn[(v, n)]

    def f(self, v, n):
        """
        frequency for pair
        f(v, n)
        """

        return self.dataset.f_vn[(v, n)]

    def train(self):
        """
        Train the algorithm
        """

        ys_v = [v for (v, n) in self.dataset.ys]
        ys_n = [n for (v, n) in self.dataset.ys]
        f_ys = np.array(self.dataset.f_ys)

        for i in range(self.current_iter, self.em_iters):
            self.current_iter = i

            likelihood = self.em_iter(ys_v, ys_n, f_ys)

            self.likelihoods.append(likelihood)
            print('%i: Log-likelihood: %f' % (i, likelihood))

            if i % 10 == 0:
                self.store()

    def em_iter(self, ys_v, ys_n, f_ys):
        """
        Do an EM step
        :return: log-likelihood
        """

        p_c_1 = np.array(self.p_c)
        p_vc_1 = np.array(self.p_vc)
        p_nc_1 = np.array(self.p_nc)

        p_c_vn = (self.p_c * self.p_vc[ys_v, :] * self.p_nc[ys_n, :]).T
        p_vn = np.sum(p_c_vn, axis=0)  # P(v, n)
        p_c_vn /= p_vn  # P(c|v, n)

        likelihood = np.sum(f_ys * np.log(p_vn))

        # d = Sigma_y f(y)p(x|y)
        d = np.sum(f_ys * p_c_vn, axis=1)

        for v in range(self.dataset.n_vs):
            # Sigma_y in {v} X N f(y)p(x|y) / d
            ys_per_v = self.dataset.ys_per_v[v]
            p_vc_1[v, :] = np.sum(f_ys[ys_per_v] * p_c_vn[:, ys_per_v], axis=1) / d

        for n in range(self.dataset.n_ns):
            # Sigma_y in N X {v} f(y)p(x|y) / d
            ys_per_n = self.dataset.ys_per_n[n]
            p_nc_1[n, :] = np.sum(f_ys[ys_per_n] * p_c_vn[:, ys_per_n], axis=1) / d

        # d / |Y|
        p_c_1 = d / self.dataset.n_ys

        self.p_c = p_c_1
        self.p_vc = p_vc_1
        self.p_nc = p_nc_1

        return likelihood

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


def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    gold_corpus = path.join(data_path, 'gold_deps.txt')
    all_pairs = path.join(data_path, 'all_pairs')

    dataset = Dataset.load(gold_corpus)

    LSCVerbClasses(dataset, n_cs=30, em_iters=50, name='gold_corpus').train()

if __name__ == "__main__":
    main()