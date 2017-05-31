
import sys
sys.path.append("..")
from main import Dataset
from os import path
import numpy as np
import pickle
from VAE import VAE
import time
import os
from sklearn.cluster import KMeans
from collections import defaultdict
import scipy.sparse as sp
from theano import sparse


np.random.seed(42)
hu_encoder = 400
hu_decoder = 400
n_latent = 20
continuous = False
n_epochs = 40

data_path = path.join(path.dirname(__file__), '../..', 'data')
gold_corpus = path.join(data_path, 'gold_deps.txt')
all_pairs = path.join(data_path, 'all_pairs')
path = "./"

def train_VAE(x_train):
    batch_order = np.arange(int(model.N / model.batch_size))

    epoch = 0
    LB_list = []

    if os.path.isfile(path + "params.pkl"):
        print("Restarting from earlier saved parameters!")
        model.load_parameters(path)
        LB_list = np.load(path + "LB_list.npy")
        epoch = len(LB_list)

    if __name__ == "__main__":
        print("iterating")
        batch_size = 100
        while epoch < n_epochs:
            epoch += 1
            start = time.time()
            np.random.shuffle(batch_order)
            LB = 0.

            for batch in batch_order:
                batch_LB = model.update( x_train[batch*batch_size:(batch+1)*batch_size].toarray() , epoch)
                LB += batch_LB

            LB /= len(batch_order)

            LB_list = np.append(LB_list, LB)
            print("Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start))
            np.save(path + "LB_list.npy", LB_list)
            model.save_parameters(path)


def create_train_matrix(dataset):
    shape_x_train = (len(dataset.ys), (len(dataset.vs) + len(dataset.ps) + len(dataset.ns)))
    start_ps = len(dataset.vs)
    start_ns = start_ps + len(dataset.ps)

    row = []
    col = []
    data = []
    for i, x in enumerate(dataset.ys):
        col.append(x[2])
        col.append(start_ps + x[3])
        col.append(start_ns + x[1])  # add noun
        # initialize other matrices
        row.append(i)
        row.append(i)
        row.append(i)
        data.append(1)
        data.append(1)
        data.append(1)

    mtx = sp.csc_matrix((data, (row, col)), shape=shape_x_train)
    return mtx

def cluster_hidden_vectors(hidden_vectors, dataset):
    kmeans = KMeans(n_clusters = 30, random_state=0 ).fit(hidden_vectors)
    print(kmeans.labels_)
    print(len(kmeans.labels_))

    clusters = defaultdict(lambda: defaultdict(int))

    for i, label in enumerate(kmeans.labels_):

        key = dataset.vs[ dataset.train_VAE[i][0] ]
        print(label)
        clusters[label][ key ]+= 1

    return dict(clusters)



if __name__ == "__main__":


    dataset = Dataset.load(gold_corpus, n_test_pairs=3000)
    print("dataset created")

    print("create train matrix")
    x_vae = create_train_matrix(dataset)
    print("instantiating model")

    model = VAE(continuous, hu_encoder, hu_decoder, n_latent, x_vae, len(dataset.vs), len(dataset.ps), len(dataset.ns))


    trainModel = True
    if(trainModel):
        train_VAE(x_vae)
    else:
        model.load_parameters(path)


    # Determine Hidden factors
    # z =  model.encode(x_vae)

    # y = model.decode(z)
    #
    # clusters2 = cluster_hidden_vectors(z[0], dataset)
    # pickle.dump(clusters2 , open( "clusters.pkl", "wb" ))


    # verb_range = [0,len(dataset.vs)-1]
    # verb_suffix_range =  [verb_range[1] +1, verb_range[1] + len(dataset.vps) ]
    # noun_range = [verb_suffix_range[1]+ 1, verb_suffix_range[1] + len(dataset.ns) ]
    #
    #
    # print( dataset.train_VAE[0])
    # print("verb_only", np.argmax( y[0][0][verb_range[0]: verb_range[1]] + 1 ))
    # print("verb_suffix", np.argmax( y[0][0][verb_suffix_range[0]: verb_suffix_range[1] + 1] ))
    # print("noun", np.argmax(y[0][0][noun_range[0]: noun_range[1] + 1]))





    # print("lala")
    # print(len(z))
    # print(x_train[0])
    # print(len(z[0][0]))
    # print(z[0][0])
    # print("y=e")
    # print(len(x))
    # print(len(y[0][0]))

    # print(verb_range)
    # print(verb_suffix_range)
    # print(noun_range)

    # reconstructed, logpxz = model.decoder(x,z)







