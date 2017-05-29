# coding: utf-8

import random
import numpy as np
import matplotlib
from sklearn.manifold import TSNE
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class EvaluationEmbeddingCentroids:
    """
    Evaluation using embeddings and cluster centroids in embedding space
    """

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.cv_emb = np.zeros((self.model.n_cs, 300))
        self.cn_emb = np.zeros((self.model.n_cs, 300))
        self.cy_emb = np.zeros((self.model.n_cs, 600))

        self.vs_emb = [emb for i, (v, emb) in enumerate(self.dataset.vs_emb.items()) if i < 250]
        self.ns_emb = [emb for i, (n, emb) in enumerate(self.dataset.ns_emb.items()) if i < 250]
        self.ys_emb = list()

        random.seed(1)
        for (vp, n, v, _) in random.sample(self.dataset.ys, 500):

            if len(self.ys_emb) >= 250:
                break

            if n in self.dataset.ns_emb and v in self.dataset.vs_emb:
                self.ys_emb.append(
                    np.concatenate(
                        (self.dataset.ns_emb[n], self.dataset.vs_emb[v]),
                        axis=0
                    )
                )

    def evaluate(self, file_name='tsne'):

        self.find_verb_cluster_embeddings()
        self.tsne(self.cv_emb, self.vs_emb, file_name='%s-verb' % file_name, color='r')

        self.find_noun_cluster_embeddings()
        self.tsne(self.cn_emb, self.ns_emb, file_name='%s-noun' % file_name, color='b')

        self.find_cluster_embeddings()
        self.tsne(self.cy_emb, self.ys_emb, file_name='%s-pairs' % file_name, color='g')

    def find_verb_cluster_embeddings(self):

        for c in range(self.model.n_cs):

            for vp in range(self.dataset.n_vps):

                w = self.model.p_vc[vp, c]
                (vt, pt) = self.dataset.vps[vp]
                v = self.dataset.vs_dict[vt]

                if v in self.dataset.vs_emb:
                    self.cv_emb[c, :] += w * self.dataset.vs_emb[v]

    def find_noun_cluster_embeddings(self):

        for c in range(self.model.n_cs):

            for n in range(self.dataset.n_ns):

                w = self.model.p_nc[n, c]

                if n in self.dataset.ns_emb:
                    self.cn_emb[c, :] += w * self.dataset.ns_emb[n]

    def find_cluster_embeddings(self):

        p_c_vn, p_vn = self.model.p_c_vn()

        for c in range(self.model.n_cs):

            for y, (vp, n, v, _) in enumerate(self.dataset.ys):

                if n in self.dataset.ns_emb and v in self.dataset.vs_emb:

                    self.cy_emb[c, :] += p_c_vn[c, y] * np.concatenate(
                        (self.dataset.ns_emb[n], self.dataset.vs_emb[v]),
                        axis=0
                    )

    def tsne(self, c_emb, v_emb, file_name='tsne', color='r'):

        try:
            colors = ['k', color]
            markers = ['x', 'o']

            to_fit = np.asarray(v_emb + c_emb.tolist())

            tsne = TSNE(n_components=2, random_state=0)
            feature_projections = tsne.fit_transform(to_fit)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            y_is = [0] * len(v_emb) + [1] * len(c_emb)

            for y_i, (x, y) in zip(y_is, feature_projections.tolist()):
                ax.scatter(x, y, c=colors[y_i], marker=markers[y_i])

            ax.set_xlim(min(feature_projections[:, 0] * 2), max(feature_projections[:, 0] * 2))
            ax.set_ylim(min(feature_projections[:, 1] * 2), max(feature_projections[:, 1] * 2))

            plt.savefig('../out/tsne/%s.png' % file_name)
            plt.clf()
            plt.close()
        except:
            pass
