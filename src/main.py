# coding: utf-8
import random
from collections import defaultdict
from os import path
from sys import stdout
import numpy as np
import pickle
from nltk.stem.porter import *


"""
This file contains the entire implementation of "Inducing a Semantically Annotated Lexicon via EM-Based Clustering"

The implementation was written by:
[Maurits Bleeker](https://github.com/MBleeker)
[Thijs Scheepers](http://github.com/tscheepers)
[David Zomerdijk](https://github.com/DavidZomerdijk)

Make sure the `data` directory contains the `all_pairs` file before running.
"""


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
        print("\tUnique verb-pos tags:\t%d" % dataset.n_ps)
        print("\tUnique verb-pos combinations:\t%d" % dataset.n_vps)
        print("\tUnique nouns:\t%d" % dataset.n_ns)
        print("\tUnique verb-noun pairs (train):\t%d" % dataset.n_ys)
        print("\tUnique verb-noun pairs (test):\t%d" % dataset.n_ys_test)
        print("\tUnique intransitive subjects:\t%d" % len(dataset.ns_in_subj))
        print("\tUnique transitive subjects:\t\t%d" % len(dataset.ns_tr_subj))
        print("\tUnique transitive objects:\t\t%d" % len(dataset.ns_tr_obj))
        print("\tUnique subject-object pairs:\t%d" % len(dataset.ws))
        print("\tEmbeddings for verbs:\t%d" % len(dataset.vs_emb))
        print("\tEmbeddings for nouns:\t%d" % len(dataset.ns_emb))

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

    def __init__(self, file_path, max_lines=None, n_test_pairs=0, embedding_file_path='../data/glove.840B.300d.txt'):
        """
        Initialize a Dataset and read it in
        """

        lines = self.read_lines(file_path)

        self.n_test_pairs = n_test_pairs

        # Token-index lookup
        self.vps = list()
        self.vps_dict = dict()
        self.f_vp = list()

        self.vs = list()
        self.vs_dict = dict()
        self.f_v = list()

        self.ps = list()
        self.ps_dict = dict()
        self.f_p = list()

        self.ns = list()
        self.ns_dict = dict()
        self.f_n = list()

        self.ns_in_subj = list()
        self.ns_in_subj_dict = dict()
        self.f_in_subj = list()

        self.ns_tr_subj = list()
        self.ns_tr_subj_dict = dict()
        self.ns_tr_obj = list()
        self.ns_tr_obj_dict = dict()

        self.ys = list()  # v,n pairs
        self.ys_dict = dict()
        self.f_ys = list()  # frequencies

        self.ws = list()  # n, n pairs
        self.ws_dict = dict()
        self.f_ws = list()

        self.ys_per_vp = defaultdict(list)
        self.ys_per_n = defaultdict(list)

        self.ys_test = list()  # v,n pairs
        self.ys_test_dict = dict()
        self.f_ys_test = list()  # frequencies

        self.vs_per_stem = defaultdict(list)
        self.unstemmed_vs = dict()

        # We will filter these auxiliary verbs
        self.aux_vts = [
            'am', 'are', 'is', 'was', 'were', 'being',
            'been', 'be', 'can', 'could', 'dare', 'do',
            'does', 'did', 'have', 'has', 'had', 'having',
            'may', 'might', 'must', 'need', 'ought',
            'shall', 'should', 'will', 'would',
        ]

        # We will stem verbs
        self.stemmer = PorterStemmer()

        if max_lines is None or max_lines > len(lines):
            n_lines = len(lines)
        else:
            n_lines = max_lines

        prev_n_tr_subj = None

        for i, ln in enumerate(lines):

            if i > n_lines:
                break

            if i % 1000 == 0:
                stdout.write("\rReading corpus… %6.2f%%" % ((100 * i) / float(n_lines),))
                stdout.flush()

            if len(ln) == 2:  # Only consider lines with two words on them, e.g. "take_so_dobj you"
                prev_n_tr_subj = self.process_line(ln, i, n_lines, prev_n_tr_subj)

        self.n_vps = len(self.vps)
        self.n_vs = len(self.vs)
        self.n_ps = len(self.ps)
        self.n_ns = len(self.ns)
        self.n_ys = len(self.ys)
        self.n_ys_test = len(self.ys_test)

        print("\rDataset read")

        emb = self.read_embeddings(embedding_file_path)
        self.vs_emb = dict()
        self.ns_emb = dict()

        for vt, ust_vts in self.vs_per_stem.items():
            # we find a weighted embedding for a stem using a weighted mean for all occurrences
            e = [emb[ust_vt] for ust_vt in ust_vts if ust_vt in emb]

            if len(e) > 0:
                v = self.vs_dict[vt]
                self.vs_emb[v] = np.mean(e, axis=1)

        for n, nt in enumerate(self.ns):
            if nt in emb:
                self.ns_emb[n] = emb[nt]

        self.store(file_path, max_lines)

    def preprocess_line(self, ln, lowercase=True, stem_verbs=True, lump_digits=True, sep='_'):

        vpt, nt = ln

        # We split the vpt: "earning_so_nsubj" into vt: "earning" and pt: "so_nsubj"
        vpt_s = vpt.split(sep)
        pt = sep.join(vpt_s[-2:])
        vt = sep.join(vpt_s[:-2])

        if lowercase:
            vt = vt.lower()

        # We stem "earning" to "earn"
        # Save unstemmed versions for later embedding based evaluation
        if stem_verbs and self.stemmer is not None:
            unstemmed_vt = vt
            stemmed_vt = self.stemmer.stem(unstemmed_vt)
            vt = stemmed_vt

            if unstemmed_vt not in self.aux_vts and stemmed_vt not in self.aux_vts:
                self.unstemmed_vs[unstemmed_vt] = stemmed_vt
                self.vs_per_stem[stemmed_vt].append(unstemmed_vt)


        vpt = (vt, pt)

        if lowercase:
            nt = nt.lower()

        # We replace all digits with DIGIT and treat them equally
        if lump_digits and nt.isdigit():
            nt = 'DIGIT'

        return vpt, nt, vt, pt

    def process_line(self, ln, i, n_lines, prev_n_tr_subj=None):

        is_train = i < n_lines - self.n_test_pairs

        vpt, nt, vt, pt = self.preprocess_line(ln)

        if vt in self.aux_vts:
            return None

        # -------------------------
        # Datastructures for step 1
        # -------------------------

        if nt not in self.ns_dict:
            n = len(self.ns)
            self.ns.append(nt)
            self.ns_dict[nt] = n
            self.f_n.append(1 if is_train else 0)
        else:
            n = self.ns_dict[nt]
            if is_train: self.f_n[n] += 1

        if vpt not in self.vps_dict:
            vp = len(self.vps)
            self.vps.append(vpt)
            self.vps_dict[vpt] = vp
            self.f_vp.append(1 if is_train else 0)
        else:
            vp = self.vps_dict[vpt]
            if is_train: self.f_vp[vp] += 1

        if vt not in self.vs_dict:
            v = len(self.vs)
            self.vs.append(vt)
            self.vs_dict[vt] = v
            self.f_v.append(1 if is_train else 0)
        else:
            v = self.vs_dict[vt]
            if is_train: self.f_v[v] += 1

        if pt not in self.ps_dict:
            p = len(self.ps)
            self.ps.append(pt)
            self.ps_dict[pt] = p
            self.f_p.append(1 if is_train else 0)
        else:
            p = self.ps_dict[pt]
            if is_train: self.f_p[p] += 1

        if is_train:
            y = self.process_pair(n, vp, self.ys, self.ys_dict, self.f_ys)

            if y not in self.ys_per_vp[vp]:
                self.ys_per_vp[vp].append(y)
            if y not in self.ys_per_n[n]:
                self.ys_per_n[n].append(y)
        else:
            self.process_pair(n, vp, self.ys_test, self.ys_test_dict, self.f_ys_test)

        # -------------------------
        # Datastructures for step 2
        # -------------------------

        if is_train:

            n_tr_obj = None
            n_tr_subj = prev_n_tr_subj

            # Subject of an intransitive verb
            if pt == 's_nsubj':
                if nt not in self.ns_in_subj_dict:
                    n_in_subj = n
                    self.ns_in_subj_dict[nt] = (n_in_subj, len(self.ns_in_subj))
                    self.ns_in_subj.append(n_in_subj)
                    self.f_in_subj.append(1)
                else:
                    n_in_subj, n_in_subj_i = self.ns_in_subj_dict[nt]
                    self.f_in_subj[n_in_subj_i] += 1

            # Subject of an transitive verb
            elif pt == 'so_nsubj':
                if nt not in self.ns_tr_subj_dict:
                    n_tr_subj = n
                    self.ns_tr_subj_dict[nt] = n_tr_subj
                    self.ns_tr_subj.append(n_tr_subj)
                else:
                    n_tr_subj = self.ns_tr_subj_dict[nt]

            # Object of an transitive verb
            elif pt == 'so_dobj':
                if nt not in self.ns_tr_obj_dict:
                    n_tr_obj = n
                    self.ns_tr_obj_dict[nt] = n_tr_obj
                    self.ns_tr_obj.append(n_tr_obj)
                else:
                    n_tr_obj = self.ns_tr_obj_dict[nt]

            if n_tr_obj is not None and n_tr_subj is not None:

                wp = (n_tr_subj, n_tr_obj)

                if wp not in self.ws_dict:
                    w = len(self.ws)
                    self.ws.append(wp)
                    self.f_ws.append(1)
                    self.ws_dict[wp] = w
                else:
                    w = self.ws_dict[wp]
                    self.f_ws[w] += 1

            return n_tr_subj

    def process_pair(self, n, vp, ys, ys_dict, f_ys):

        yp = (vp, n)

        if yp not in ys_dict:
            y = len(ys)
            ys.append(yp)
            f_ys.append(1)
            ys_dict[yp] = y
        else:
            y = ys_dict[yp]
            f_ys[y] += 1

        return y

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

    def read_embeddings(self, embedding_file_path, vector_length=300):
        """
        Load the pretrained vectors
        """
        embeddings = dict()

        with open(embedding_file_path, 'r') as f:
            for i, ln in enumerate(f):

                if i % 1000 == 0:
                    stdout.write("\rReading embeddings… %d\tfrom %s" % (i, embedding_file_path))

                ln = ln.split()
                if len(ln) == vector_length + 1:
                    token = ln[0]
                    if token in self.unstemmed_vs or token in self.ns:
                        embedding = np.array([float(x) for x in ln[1:]])
                        embeddings[token] = embedding

        print("\rEmbeddings read from %s" % (embedding_file_path,))

        return embeddings


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

        ys_v = [v for (v, n) in self.dataset.ys]
        ys_n = [n for (v, n) in self.dataset.ys]
        f_ys = np.array(self.dataset.f_ys)

        evaluator = EvaluationPseudoDisambiguation(self.dataset, self, lower_bound=30)

        for i in range(self.current_iter, self.em_iters):
            self.current_iter = i

            likelihood = self.em_iter(ys_v, ys_n, f_ys)

            self.likelihoods.append(likelihood)

            if i % 5 == 0:
                acc = evaluator.evaluate()
                self.accuracies[i] = acc
                print('%i: Log-likelihood: %f\tAccuracy:\t%f' % (i, likelihood, acc))
            else:
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

        for v in range(self.dataset.n_vps):
            # Sigma_y in {v} X N f(y)p(x|y) / d
            ys_per_v = self.dataset.ys_per_vp[v]
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


class SubjectIntransitiveVerbClasses:
    """
    Clustering for subjects for fixed intransitive verbs
    This is the implementation of Step 2
    Verb classes (corresponding to Section 4.1 in the paper https://arxiv.org/abs/cs/9905008)
    """

    def __init__(self, dataset, model, em_iters=50, name='step2-1'):
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

        p_c = np.random.rand(self.model.n_cs)
        p_c /= np.sum(p_c)

        return p_c

    def train(self):
        """
        Train the algorithm
        """

        fs = np.array(self.dataset.f_in_subj)

        for i in range(self.em_iters):
            self.current_iter = i

            likelihood = self.em_iter(fs)
            self.likelihoods.append(likelihood)

            print('%i: Log-likelihood: %f' % (i, likelihood))

            if i % 10 == 0:
                self.store()

    def em_iter(self, fs):
        """
        Do an EM step
        :return: log-likelihood
        """

        p_c_n = (self.p_c * self.model.p_nc[self.dataset.ns_in_subj, :]).T # p(c)P_LC(n|c)
        p_n = np.sum(p_c_n, axis=0)  # P(n)
        p_c_n /= p_n  # P(c|n)

        likelihood = np.sum(fs * np.log(p_n))

        self.p_c = np.sum(fs * p_c_n, axis=1) / np.sum(fs)

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


def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    gold_corpus = path.join(data_path, 'gold_deps.txt')
    all_pairs = path.join(data_path, 'all_pairs')

    dataset = Dataset.load(gold_corpus, n_test_pairs=10)

    # Parameter grid
    parameters = [
        (5, 151),
        (10, 151),
        (20, 151),
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