# coding: utf-8
from collections import defaultdict
from os import path

import numpy as np


def read_corpus(path):
    """Read a file as a list of tuples"""

    with open(path, 'r') as f:
        return [tuple(ln.strip().split()) for ln in f]

def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    gold_corpus = path.join(data_path, 'gold_deps.txt')
    all_pairs = path.join(data_path, 'all_pairs')
    corpus = read_corpus(gold_corpus)

    ns_per_v = defaultdict(list)
    vs_per_n = defaultdict(list)
    f_vn = defaultdict(int)
    ys = []

    for vt, nt in corpus:
        ns_per_v[vt] += [nt]
        vs_per_n[nt] += [vt]
        ys += [(vt, nt)]
        f_vn[(vt, nt)] += 1

    # Number of classes
    n_cs = 30
    ys = list(set(ys))
    ns = list(set(vs_per_n.keys()))
    vs = list(set(ns_per_v.keys()))
    n_unique_vs = len(vs)
    n_unique_ns = len(ns)
    n_unique_ys = len(ys)

    # Theta, parameters
    p_c = np.random.rand(n_cs)
    p_c /= np.sum(p_c)
    p_vc = np.random.rand(n_unique_vs, n_cs)
    p_vc /= np.sum(p_vc, axis=0)
    p_nc = np.random.rand(n_unique_ns, n_cs)
    p_nc /= np.sum(p_nc, axis=0)

    def p_c_vn(c, v, n):

        if isinstance(v, str):
            v = vs.index(v)

        if isinstance(n, str):
            n = ns.index(n)

        return p_c[c] * p_vc[v, c] * p_nc[n, c] / p_vn(v, n)

    def p_vn(v, n):

        return sum([p_c[c] * p_vc[v, c] * p_nc[n, c] for c in range(n_cs)])

    def f(v, n):

        if isinstance(v, int):
            v = vs[v]

        if isinstance(n, int):
            n = ns[n]

        return f_vn[(v, n)]

    # EM iterations
    for i in range(10):
        p_c_1 = np.array(p_c)
        p_vc_1 = np.array(p_vc)
        p_nc_1 = np.array(p_nc)

        likelihood = 0

        for c in range(n_cs):

            # Sigma_y f(y)p(x|y)
            d = sum([f(v, n) * p_c_vn(c, v, n) for (v, n) in ys])
            likelihood += np.log(d)

            for v, vt in enumerate(vs):
                p_vc_1[v, c] = sum([f(v, n) * p_c_vn(c, v, n) for n in ns_per_v[vt]]) / d

            for n, nt in enumerate(ns):
                p_nc_1[n, c] = sum([f(v, n) * p_c_vn(c, v, n) for v in vs_per_n[nt]]) / d

            p_c_1[c] = d

        print('Likelihood: %f', likelihood)

        p_c_1 /= n_unique_ys

        p_c = p_c_1
        p_vc = p_vc_1
        p_nc = p_nc_1


if __name__ == "__main__":
    main()