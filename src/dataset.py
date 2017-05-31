# coding: utf-8

from collections import defaultdict
from os import path
from sys import stdout
import numpy as np
import pickle
from nltk.stem.porter import *


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
        self.vps = list()  # Lookup for unique verb-pos combination
        self.vps_dict = dict()  # Reverse lookup
        self.f_vp = list()  # Frequencies

        self.vs = list()  # Lookup for unique verb
        self.vs_dict = dict()  # Reverse lookup
        self.f_v = list()  # Frequencies

        self.ps = list()  # Lookup for unique pos-tag of verb
        self.ps_dict = dict()  # Reverse lookup
        self.f_p = list()  # Frequencies

        self.ns = list()  # Lookup for unique noun
        self.ns_dict = dict()  # Reverse lookup
        self.f_n = list()  # Frequencies

        self.ns_in_subj = list()  # Lookup for unique noun as subject for intransitive verb
        self.ns_in_subj_dict = dict()  # Reverse lookup
        self.f_in_subj = list()  # Frequencies

        self.ns_tr_subj = list()  # Lookup for unique noun as subject for transitive verb
        self.ns_tr_subj_dict = dict()  # Reverse lookup
        self.ns_tr_obj = list()  # Lookup for unique noun as object for transitive verb
        self.ns_tr_obj_dict = dict()  # Reverse lookup

        self.ys = list()  # vp,n,v,p pairs
        self.ys_dict = dict()  # Reverse lookup
        self.f_ys = list()  # Frequencies

        self.ws = list()  # n, n pairs
        self.ws_dict = dict()  # Reverse lookup
        self.f_ws = list()  # Frequencies

        self.ys_per_vp = defaultdict(list)
        self.ys_per_n = defaultdict(list)

        self.ys_test = list()  # vp,n,v,p pairs
        self.ys_test_dict = dict()  # Reverse lookup
        self.f_ys_test = list()  # Frequencies

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
        self.vs_emb = dict()  # Embeddings for stemmed unique verbs
        self.ns_emb = dict()  # Embeddings for unique nouns

        for vt, ust_vts in self.vs_per_stem.items():
            # We find a weighted embedding for a stem using a weighted mean for all occurrences
            e = [emb[ust_vt] for ust_vt in ust_vts if ust_vt in emb]

            if len(e) > 0:
                v = self.vs_dict[vt]
                self.vs_emb[v] = np.mean(e, axis=0)

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
            y = self.process_pair(n, vp, v, p, self.ys, self.ys_dict, self.f_ys)

            if y not in self.ys_per_vp[vp]:
                self.ys_per_vp[vp].append(y)
            if y not in self.ys_per_n[n]:
                self.ys_per_n[n].append(y)
        else:
            self.process_pair(n, vp, v, p, self.ys_test, self.ys_test_dict, self.f_ys_test)

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

    def process_pair(self, n, vp, v, p, ys, ys_dict, f_ys):

        yp = (vp, n, v, p)

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
                    t = ln[0]
                    if t in self.unstemmed_vs or t in self.ns_dict:
                        embedding = np.array([float(x) for x in ln[1:]])
                        embeddings[t] = embedding

        print("\rEmbeddings read from %s" % (embedding_file_path,))

        return embeddings