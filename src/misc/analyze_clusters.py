from __future__ import division
from __future__ import print_function
import os.path
import random
from dataset import Dataset
import numpy as np
from collections import defaultdict
import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
gold_corpus = os.path.join(data_path, 'gold_deps.txt')
all_pairs = os.path.join(data_path, 'all_pairs')

#Todo: get the most recent dataset
dataset = Dataset.load(all_pairs, n_test_pairs=3000)

v_dim = dataset.n_vs
n_dim = dataset.n_ns
p_dim = dataset.n_ps

input_dim = v_dim + n_dim + p_dim
hidden_encoder_dim = 600
hidden_decoder_dim = 600
latent_dim = 30
lam = 0.01

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

V = tf.placeholder("int32", shape=[None])
N = tf.placeholder("int32", shape=[None])
P = tf.placeholder("int32", shape=[None])

v = tf.one_hot(V, v_dim)
n = tf.one_hot(N, n_dim)
p = tf.one_hot(P, p_dim)

x = tf.concat([v, n, p], 1)

l2_loss = tf.constant(0.0)

W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.multiply(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction

v_logits, n_logits, p_logits = tf.split(x_hat, [v_dim, n_dim, p_dim], 1)

v_sce = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=v_logits, labels=V))
n_sce = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits, labels=N))
p_sce = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits, labels=P))

BCE = v_sce + n_sce + p_sce

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_epochs = 1
batch_size = 100

ys = dataset.ys
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_latent_vectors():
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('../out/tf/experiment',
                                              graph=sess.graph)
        if os.path.isfile("../out/model-2.ckpt"):
            print("Restoring saved parameters")
            saver.restore(sess, "../out/model-2.ckpt")
        else:
            print("Initializing parameters")
            sess.run(tf.global_variables_initializer())

        step = 0


        total_output = []
        print(len(list(chunks(ys, batch_size))))
        for batch in list(chunks(ys, batch_size)):
            _, ns, vs, ps = [list(t) for t in zip(*batch)]
            step += 1
            feed_dict = {V: vs, N: ns, P: ps}
            output = sess.run(z, feed_dict=feed_dict)
            total_output.append(output)
            print(step)

        latent_vectors = np.vstack(total_output)
        pickle.dump(latent_vectors, open("../out/latent_vectors.p", "wb"))
        return latent_vectors



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def normalize(x):
    x = np.array(x)
    return x / x.max(axis=0)

def determine_ys_cluster(l_vectors, ys):
    ys_clusters = defaultdict(lambda: defaultdict(int))
    for  i,(vp,n,v,p) in enumerate(ys):
        ys_clusters[int(l_vectors[i].argmax(axis=0))][v] += 1


    return ys_clusters


def create_final_clusters(clusters, dataset, use_softmax = True):
    # Now determine which class occurs most often
    final_clusters = defaultdict(list)

    for verb in dataset.vs:
        counts = []

        for c in range(len(clusters)):

            if dataset.vs_dict[verb] in clusters[c]:
                counts.append(clusters[c][dataset.vs_dict[verb]])
            else:
                counts.append(0)

        if use_softmax:
            softmax_list = softmax(counts)
            cluster = int(softmax_list.argmax(axis=0))
            if isinstance(cluster, int):
                final_clusters[cluster].append((verb, softmax_list[cluster]))
        else:
            norm_list = normalize(counts)
            cluster = int(norm_list.argmax(axis=0))
            if isinstance(cluster, int):
                final_clusters[cluster].append((verb, norm_list[cluster]))

    return final_clusters

#print top n of clusters
def print_clusters(clusters, n):
    for i in range(len(clusters)):
        print("Cluster: %d" % (i))
        for line in sorted(clusters[i], key=lambda tup: tup[1],reverse=True)[:n]:
            print(line  )




if __name__ == "__main__":
    make_new_latent_vectors = False

    if make_new_latent_vectors:
        latent_vectors = get_latent_vectors()
    else:
        latent_vectors = pickle.load(open("../out/latent_vectors.p", "rb"))

    clusters = create_final_clusters(determine_ys_cluster(latent_vectors, ys), dataset)
    pickle.dump(clusters, open("../out/clusters.p", "wb"))

    print_clusters(clusters, 6)
