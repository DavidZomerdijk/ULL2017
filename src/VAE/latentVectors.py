
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

def train_VAE():
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
        while epoch < n_epochs:
            epoch += 1
            start = time.time()
            np.random.shuffle(batch_order)
            LB = 0.

            for batch in batch_order:
                batch_LB = model.update(batch, epoch)
                LB += batch_LB

            LB /= len(batch_order)

            LB_list = np.append(LB_list, LB)
            print("Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start))
            np.save(path + "LB_list.npy", LB_list)
            model.save_parameters(path)

def create_train_matrix(dataset):
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    mtx = sparse.csc_matrix((data, (row, col)), shape=(3, 3))

    # x_train = []
    # for i in dataset.train_VAE:
    #     verb_temp = np.zeros(len(dataset.vs))
    #     verb_temp.itemset(i[0],int(1))
    #
    #     verb_suf_temp = np.zeros(len(dataset.vps))
    #     verb_suf_temp.itemset(i[1],int(1))
    #
    #     noun_temp = np.zeros(len(dataset.ns))
    #     noun_temp.itemset(i[2],int(1))
    #
    #     x_train.append( np.concatenate((verb_temp, verb_suf_temp, noun_temp)))
    # return np.array(x_train)

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


    dataset = Dataset.load(all_pairs, n_test_pairs=3000)
    print("dataset created")
    X_VAE = create_train_matrix(dataset)

    print("instantiating model")

    model = VAE(continuous, hu_encoder, hu_decoder, n_latent, X_VAE)

    trainModel = False
    if(trainModel):
        train_VAE()
    else:
        model.load_parameters(path)


    # Determine Hidden factors
    z =  model.encode(X_VAE)

    # y = model.decode(z)

    clusters2 = cluster_hidden_vectors(z[0], dataset)
    pickle.dump(clusters2 , open( "clusters.pkl", "wb" ))


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







