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
    def load(cls, file_path, max_lines=None, n_test_pairs=0):
        """
        Loads .pkl file if available, otherwise creates dataset
        :return: Dataset
        """

        pickle_path = cls.pickle_path(file_path, max_lines=max_lines, n_test_pairs=n_test_pairs)

        if path.isfile(pickle_path):
            print("Loading dataset (%s) from disk" % pickle_path)
            dataset = pickle.load(open(pickle_path, 'rb'))
        else:
            dataset = Dataset(file_path, max_lines=max_lines, n_test_pairs=n_test_pairs)

        print("Dataset ready")
        print("\tUnique verbs:\t%d" % dataset.n_vs)
        print("\tUnique nouns:\t%d" % dataset.n_ns)
        print("\tUnique pairs (train):\t%d" % dataset.n_ys_train)
        print("\tUnique pairs (test):\t%d" % dataset.n_ys_test)

        return dataset

    @classmethod
    def pickle_path(cls, file_path, max_lines=None, n_test_pairs=0):
        """
        Returns the path of the pickle file
        """

        if max_lines is not None:
            file_path += '-l%d' % max_lines

        if n_test_pairs != 0:
            file_path += '-t%d' % n_test_pairs

        return file_path + '.pkl'

    def __init__(self, file_path, max_lines=None, n_test_pairs=0):
        """
        Initialize a Dataset and read it in
        """

        lines = self.read_lines(file_path)

        self.n_test_pairs = n_test_pairs

        # Token-index lookup
        self.ns = list()
        self.ns_dict = dict()
        self.vs = list()
        self.vs_dict = dict()

        self.f_n_train = list()
        self.f_v_train = list()

        self.ys_train = list()  # v,n pairs
        self.ys_train_dict = dict()
        self.f_ys_train = list()  # frequencies

        self.ys_train_per_v = defaultdict(list)
        self.ys_train_per_n = defaultdict(list)

        self.ys_test = list()  # v,n pairs
        self.ys_test_dict = dict()
        self.f_ys_test = list()  # frequencies

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
                self.process_line(ln, i, n_lines)

        self.n_vs = len(self.vs)
        self.n_ns = len(self.ns)
        self.n_ys_train = len(self.ys_train)
        self.n_ys_test = len(self.ys_test)

        print("\rDataset read")

        self.store(file_path, max_lines)

    def process_line(self, ln, i, n_lines):

        vt, nt = ln
        is_train = i < n_lines - self.n_test_pairs

        # -------------------------
        # Datastructures for step 1
        # -------------------------

        if nt not in self.ns_dict:
            n = len(self.ns)
            self.ns.append(nt)
            self.ns_dict[nt] = n
            if is_train: self.f_n_train.append(1)
        else:
            n = self.ns_dict[nt]
            if is_train: self.f_n_train[n] += 1

        if vt not in self.vs_dict:
            v = len(self.vs)
            self.vs.append(vt)
            self.vs_dict[vt] = v
            if is_train: self.f_v_train.append(1)
        else:
            v = self.vs_dict[vt]
            if is_train: self.f_v_train[v] +=1

        if is_train:
            y = self.process_pair(n, v, self.ys_train, self.ys_train_dict, self.f_ys_train)

            if y not in self.ys_train_per_v[v]:
                self.ys_train_per_v[v].append(y)
            if y not in self.ys_train_per_n[n]:
                self.ys_train_per_n[n].append(y)
        else:
            self.process_pair(n, v, self.ys_test, self.ys_test_dict, self.f_ys_test)

    def process_pair(self, n, v, ys, ys_dict, f_ys):

        yp = (v, n)

        if yp not in ys_dict:
            y = len(ys)
            ys.append(yp)
            f_ys.append(1)
            ys_dict[yp] = y
        else:
            y = ys_dict[yp]
            f_ys[y] += 1

        return y

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.ns, self.vs, self.f_n_train, self.f_v_train, self.f_ys_train, self.ys_train, self.ys_train_per_v, self.ys_train_per_n, \
               self.f_ys_test, self.ys_test, self.n_vs, self.n_ns, self.n_ys_train, self.n_ys_test

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.ns, self.vs, self.f_n_train, self.f_v_train, self.f_ys_train, self.ys_train, self.ys_train_per_v, self.ys_train_per_n, self.f_ys_test, \
        self.ys_test, self.n_vs, self.n_ns, self.n_ys_train, self.n_ys_test = state

    def store(self, file_path, max_lines):
        """
        Stores dataset to .pkl file
        """
        pickle.dump(self, open(Dataset.pickle_path(file_path, max_lines=max_lines, n_test_pairs=self.n_test_pairs), 'wb'))

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

    def train(self):
        """
        Train the algorithm
        """

        ys_v = [v for (v, n) in self.dataset.ys_train]
        ys_n = [n for (v, n) in self.dataset.ys_train]
        f_ys = np.array(self.dataset.f_ys_train)

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
            ys_per_v = self.dataset.ys_train_per_v[v]
            p_vc_1[v, :] = np.sum(f_ys[ys_per_v] * p_c_vn[:, ys_per_v], axis=1) / d

        for n in range(self.dataset.n_ns):
            # Sigma_y in N X {v} f(y)p(x|y) / d
            ys_per_n = self.dataset.ys_train_per_n[n]
            p_nc_1[n, :] = np.sum(f_ys[ys_per_n] * p_c_vn[:, ys_per_n], axis=1) / d

        # d / |Y|
        p_c_1 = d / self.dataset.n_ys_train

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

    dataset = Dataset.load(all_pairs, n_test_pairs=3000)

    parameters = [# (1, 101),
                  # (10, 101),
                  (20, 101),
                  (30, 101),
                  (40, 101),
                  (50, 101),
                  (60, 101),
                  (70, 101),
                  (80, 101),
                  (90, 101),
                  (100, 101)]

    for (n_cs, em_itters) in parameters:
        print("------")
        LSCVerbClasses(dataset, n_cs=n_cs, em_iters=em_itters, name='all_pairs').train()

if __name__ == "__main__":
    main()