from __future__ import division
from __future__ import print_function
import os.path
import random
import pickle
from dataset import Dataset

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
gold_corpus = os.path.join(data_path, 'gold_deps.txt')
all_pairs = os.path.join(data_path, 'all_pairs')

dataset = Dataset.load(all_pairs, n_test_pairs=10000)

class VAEVerbClasses:
    """
    Latent Semantic Clustering for Verb Classes

    This is the implementation of Step 1
    Verb classes (corresponding to Section 2 in the paper https://arxiv.org/abs/cs/9905008)
    """

    def __init__(self, dataset, hidden_dim=600, latent_dim=50, lam=0.0001):
        """
        :param dataset: The dataset for which to train
        :param n_cs: Number of classes
        :param em_iters: Iterations
        """
        self.dataset = dataset

        v_dim = dataset.n_vs
        n_dim = dataset.n_ns
        p_dim = dataset.n_ps

        input_dim = v_dim + n_dim + p_dim

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.001)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0., shape=shape)
            return tf.Variable(initial)

        self.include_VP = tf.placeholder("int32", shape=[1])
        self.V = tf.placeholder("int32", shape=[None])
        self.N = tf.placeholder("int32", shape=[None])
        self.P = tf.placeholder("int32", shape=[None])

        v = tf.cond(self.include_VP > 0, lambda: tf.one_hot(self.V, v_dim), lambda: tf.zeros(v_dim))
        n = tf.one_hot(self.N, n_dim)
        p = tf.cond(self.include_VP > 0, lambda: tf.one_hot(self.P, p_dim), lambda: tf.zeros(p_dim))

        x = tf.concat([v, n, p], 1)

        l2_loss = tf.constant(0.0)

        W_encoder_input_hidden = weight_variable([input_dim, hidden_dim])
        b_encoder_input_hidden = bias_variable([hidden_dim])
        l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

        # Hidden layer encoder
        hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

        W_encoder_hidden_mu = weight_variable([hidden_dim, latent_dim])
        b_encoder_hidden_mu = bias_variable([latent_dim])
        l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

        # Mu encoder
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

        W_encoder_hidden_logvar = weight_variable([hidden_dim, latent_dim])
        b_encoder_hidden_logvar = bias_variable([latent_dim])
        l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

        # Sigma encoder
        logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

        # Sample latent variable
        std_encoder = tf.exp(0.5 * logvar_encoder)
        z = mu_encoder + tf.multiply(std_encoder, epsilon)

        W_decoder_z_hidden = weight_variable([latent_dim, hidden_dim])
        b_decoder_z_hidden = bias_variable([hidden_dim])
        l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

        # Hidden layer decoder
        hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

        W_decoder_hidden_reconstruction = weight_variable([hidden_dim, input_dim])
        b_decoder_hidden_reconstruction = bias_variable([input_dim])
        l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

        KLD = -0.5 * tf.reduce_mean(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

        x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction

        v_logits, n_logits, p_logits = tf.split(x_hat, [v_dim, n_dim, p_dim], 1)

        v_sce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=v_logits, labels=self.V))
        n_sce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits, labels=self.N))
        p_sce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits, labels=self.P))

        self.loss = tf.reduce_mean(v_sce + n_sce + p_sce + KLD)
        tf.summary.scalar("lowerbound", self.loss)

        regularized_loss = self.loss + lam * l2_loss

        self.train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

        # Prediction of the probabilities
        self.V_prediction = tf.nn.softmax(logits=v_logits)
        self.N_prediction = tf.nn.softmax(logits=n_logits)
        self.P_prediction = tf.nn.softmax(logits=p_logits)

        # Accuracy
        self.V_accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(self.V_prediction, axis=1)), self.V)
        self.N_accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(self.N_prediction, axis=1)), self.N)
        self.P_accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(self.P_prediction, axis=1)), self.P)

        # add Saver ops
        self.saver = tf.train.Saver()

    def train(self, n_epochs=10, batch_size=512):

        # add op for merging summary
        summary_op = tf.summary.merge_all()

        ys_train = dataset.ys
        ys_test = dataset.ys_test

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        with tf.Session() as sess:

            summary_writer = tf.summary.FileWriter('../out/tf/experiment', graph=sess.graph)
            self.initialize_parameters(sess)

            _, ns_test, vs_test, ps_test = [list(t) for t in zip(*ys_test)]
            feed_dict_test = {self.V: vs_test, self.N: ns_test, self.P: ps_test, self.include_VP: 1}

            v_accs = dict()
            p_accs = dict()
            n_accs = dict()
            train_losses = dict()
            test_losses = dict()

            step = 0
            for epoch in range(1, n_epochs):

                random.shuffle(ys_train)

                for batch in list(chunks(ys_train, batch_size)):
                    _, ns, vs, ps = [list(t) for t in zip(*batch)]

                    step += 1

                    feed_dict = {self.V: vs, self.N: ns, self.P: ps, self.include_VP: 1}

                    if step % 10 == 0:
                        _, train_losses[step], summary_str = sess.run([self.train_step, self.loss, summary_op], feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        print("Step {0} | Epoch {1} | Loss: {2}".format(step, epoch, train_losses[step]))
                    else:
                        sess.run([self.train_step], feed_dict=feed_dict)

                    feed_dict = {self.V: vs, self.N: ns, self.P: ps}

                    if step % 100 == 0:
                        test_losses[step], v_accs[step], n_accs[step], p_accs[step] = sess.run([
                            self.loss, self.V_accuracy,
                            self.N_accuracy, self.P_accuracy
                        ], feed_dict=feed_dict_test)
                        print("Step {0} | Epoch {1} | Test-Loss: {2} | Verb Accuracy: {3} | Noun Accuracy: {4} | POS Accuracy: {5}".format(step, epoch, test_losses[step], v_accs[step], n_accs[step], p_accs[step]))
                    else:
                        sess.run([self.train_step], feed_dict=feed_dict)

                    if step % 1000 == 0:
                        self.store(sess, (v_accs, n_accs, p_accs, test_losses, train_losses))

            self.store(sess, (v_accs, n_accs, p_accs, test_losses, train_losses))

    def p_n_v(self, n, vp):
        """
        p(n|v)
        """

        (vt, pt) = self.dataset.vps[vp]
        v = self.dataset.vs_dict[vt]
        p = self.dataset.ps_dict[pt]

        feed_dict = {self.V: [0], self.N: [n], self.P: [0], self.include_VP: 0}

        with tf.Session() as sess:
            v_pred, p_pred = sess.run([
                self.V_prediction, self.P_prediction
            ], feed_dict=feed_dict)

            return v_pred[v] * p_pred[p]

    def store(self, sess, plot_data):
        """
        Function to save the model
        :return:
        """

        pickle.dump(plot_data, open('../out/vae_results-2.pkl', 'wb'))

        self.saver.save(sess, "../out/vae-2.ckpt")

    def initialize_parameters(self, sess):

        if os.path.isfile("../out/vae-2.ckpt"):
            print("Restoring saved parameters")
            self.saver.restore(sess, "../out/vae-2.ckpt")
        else:
            print("Initializing parameters")
            sess.run(tf.global_variables_initializer())