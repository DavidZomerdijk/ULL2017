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
        print("\tUnique intransitive subjects:\t%d" % len(dataset.ns_in_subj))
        print("\tUnique transitive subjects:\t\t%d" % len(dataset.ns_tr_subj))
        print("\tUnique transitive objects:\t\t%d" % len(dataset.ns_tr_obj))

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
        self.ns_in_subj = list()
        self.ns_tr_subj = list()
        self.ns_tr_obj = list()

        self.ns_per_v = defaultdict(set)
        self.vs_per_n = defaultdict(set)
        self.f_vn = defaultdict(int)
        self.f_in_subj = defaultdict(int)
        self.f_tr_so = defaultdict(int)
        self.ys = set()  # v,n pairs
        self.tr_sos = set()  # n,n pairs

        if max_lines is None or max_lines > len(lines):
            n_lines = len(lines)
        else:
            n_lines = max_lines

        # To have correct scoping
        n_tr_subj = 0

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

                self.ns_per_v[v] |= {n}
                self.vs_per_n[n] |= {v}
                self.ys |= {(v, n)}
                self.f_vn[(v, n)] += 1

                # -------------------------
                # Datastructures for step 2
                # -------------------------

                # Subject of an intransitive verb
                if vt.endswith('s_nsubj'):
                    if nt not in self.ns_in_subj:
                        n_in_subj = len(self.ns_in_subj)
                        self.ns_in_subj.append(nt)
                    else:
                        n_in_subj = self.ns.index(nt)

                    self.f_in_subj[n_in_subj] += 1

                # Subject of an transitive verb
                elif vt.endswith('so_nsubj'):
                    if nt not in self.ns_tr_subj:
                        n_tr_subj = len(self.ns_tr_subj)
                        self.ns_tr_subj.append(nt)
                    else:
                        n_tr_subj = self.ns.index(nt)

                # Object of an transitive verb
                elif vt.endswith('so_dobj'):
                    if nt not in self.ns_tr_obj:
                        n_tr_obj = len(self.ns_tr_obj)
                        self.ns_tr_obj.append(nt)
                    else:
                        n_tr_obj = self.ns.index(nt)

                    self.f_tr_so[(n_tr_subj, n_tr_obj)] += 1
                    self.tr_sos |= {(n_tr_subj, n_tr_obj)}

        # Lengths
        self.n_vs = len(self.vs)
        self.n_ns = len(self.ns)
        self.n_ys = len(self.ys)

        print("\rDataset read")

        self.store(file_path, max_lines)

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

    # def save_model(self, file):
    #     model = dict()
    #     model["n_nc"] = self.n_cs
    #     model["em_iters"] = self.em_iters
    #
    #     pickle.dump(model, open(file, "wb"))


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
        for i in range(self.current_iter, self.em_iters):
            self.current_iter = i

            likelihood = self.em_iter()

            self.likelihoods.append(likelihood)
            print('%i: Log-likelihood: %f' % (i, likelihood))

            self.store()

    def em_iter(self):
        """
        Do an EM step
        :param i: iteration
        :return: log-likelihood
        """

        p_c_1 = np.array(self.p_c)
        p_vc_1 = np.array(self.p_vc)
        p_nc_1 = np.array(self.p_nc)

        self.p_vn = { (v, n):
                      sum([self.p_c[c] * self.p_vc[v, c] * self.p_nc[n, c] for c in range(self.n_cs)])
                      for (v, n)
                      in self.dataset.ys
                    }

        likelihood = sum([np.log(self.f(v, n) * self.p_vn[(v, n)]) for (v, n) in self.dataset.ys])

        for c in range(self.n_cs):

            # d = Sigma_y f(y)p(x|y)
            d = sum([self.f(v, n) * self.p_c_vn(c, v, n) for (v, n) in self.dataset.ys])

            for v in range(self.dataset.n_vs):
                # Sigma_y in {v} X N f(y)p(x|y) / d
                p_vc_1[v, c] = sum([self.f(v, n) * self.p_c_vn(c, v, n) for n in self.dataset.ns_per_v[v]]) / d

            for n in range(self.dataset.n_ns):
                # Sigma_y in N X {v} f(y)p(x|y) / d
                p_nc_1[n, c] = sum([self.f(v, n) * self.p_c_vn(c, v, n) for v in self.dataset.vs_per_n[n]]) / d

            p_c_1[c] = d

        # d / |Y|
        p_c_1 /= self.dataset.n_ys

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

        out_path = path.join(path.dirname(__file__), '..', 'out', '%s-%d.pkl' % (self.name, self.current_iter))
        pickle.dump(self,  open(out_path, 'wb'))


class SubjectIntransitiveVerbClasses:
    """
    Clustering for subjects for fixed intransitive verbs

    This is the implementation of Step 2
    Verb classes (corresponding to Section 4.1 in the paper https://arxiv.org/abs/cs/9905008)
    """

    def __init__(self, dataset, n_cs=30, em_iters=50):
        """
        :param dataset: The dataset for which to train
        :param n_cs: Number of classes
        :param em_iters: Iterations
        """
        self.dataset = dataset
        self.n_cs = n_cs
        self.em_iters = em_iters

        self.p_vn = None  # Calculated each EM-Iteration
        self.p_n = None # set this on none to start with
        self.p_c = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize theta parameters
        :return:
        """
        np.random.seed(1)

        p_c = np.random.rand(self.n_cs)
        p_c /= np.sum(p_c)


        return p_c


    def p_n(self):
        """
        p(n)
        :return: probability of p(n) for a given n
        """

        return None

    def p_c_n(self, c, n):
        """
        p(c|n)
        """

        return self.p_c[c] / self.p_vn[(v, n)]

    def f(self, n):
        """
        frequency for pair
        f(v, n)
        """

        return self.dataset.f_in_subj[n]

    def train(self):
        """
        Train the algorithm
        """
        for i in range(self.em_iters):
            likelihood = self.em_iter(i)
            print('%i: Log-likelihood: %f' % (i, likelihood))

    def em_iter(self):
        """
        Do an EM step
        :param i: iteration
        :return: log-likelihood
        """

        p_c_1 = np.array(self.p_c)
        p_vc_1 = np.array(self.p_vc)
        p_nc_1 = np.array(self.p_nc)

        likelihood = 0

        for c in range(self.n_cs):

            # Sigma_y f(y)p(x|y)
            d = sum([self.f(n) * self.p_c_n(c, n) for n in self.dataset.ns_in_subj])

            p_c_1[c] = d

        # d / |Y|
        p_c_1 /= self.dataset.n_ys

        self.p_c = p_c_1
        self.p_vc = p_vc_1
        self.p_nc = p_nc_1

        return likelihood


def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    gold_corpus = path.join(data_path, 'gold_deps.txt')
    all_pairs = path.join(data_path, 'all_pairs')

    dataset = Dataset.load(gold_corpus)

    LSCVerbClasses(dataset, n_cs=30, em_iters=50, name='gold_deps').train()

if __name__ == "__main__":
    main()