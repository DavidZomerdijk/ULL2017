# coding: utf-8
from collections import defaultdict
from os import path


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
    n_unique_pairs = len(set(corpus))

    for v, n in corpus:
        ns_per_v[v] += [n]
        vs_per_n[n] += [v]
        f_vn[(v, n)] += 1

    # Number of classes
    n_cs = 30
    n_unique_vs = len(set(ns_per_v.keys()))
    n_unique_ns = len(set(vs_per_n.keys()))

    # Theta, parameters
    t_c = defaultdict(float)
    t_vc = defaultdict(float)
    t_nc = defaultdict(float)

    for c in range(n_cs):
        t_c[c] = 1. / n_cs

        for v, _ in ns_per_v.items():
            t_vc[(v, c)] = 1. / n_unique_vs

        for n, _ in vs_per_n.items():
            t_nc[(n, c)] = 1. / n_unique_ns

    # EM iterations
    for i in range(10):
        t_c_1 = defaultdict(float)
        t_vc_1 = defaultdict(float)
        t_nc_1 = defaultdict(float)

        # for c in range(n_cs):
        #     t_c[c] = 1. / n_cs
        #
        #     for v, _ in ns_per_v.items():
        #         t_vc[(v, c)] = 1. / n_unique_vs
        #
        #     for n, _ in vs_per_n.items():
        #         t_nc[(n, c)] = 1. / n_unique_ns

        t_c = t_c_1
        t_vc = t_vc_1
        t_nc = t_nc_1



    print(corpus[-1])

if __name__ == "__main__":
    main()