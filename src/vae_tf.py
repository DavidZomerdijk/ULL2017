from __future__ import division
from __future__ import print_function
import os.path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

v_dim = 1000
n_dim = 20000
p_dim = 50

input_dim = v_dim + n_dim + p_dim
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

V = tf.placeholder("int", shape=[None, 1])
N = tf.placeholder("int", shape=[None, 1])
P = tf.placeholder("int", shape=[None, 1])

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

v_logits = tf.slice(x_hat, 0, v_dim)
n_logits = tf.slice(x_hat, v_dim, n_dim)
p_logits = tf.slice(x_hat, v_dim + n_dim, p_dim)

v_sce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=v_logits, labels=v)
n_sce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits, labels=n)
p_sce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits, labels=p)

BCE = tf.reduce_sum(v_sce + n_sce + p_sce, reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_steps = int(1e6)
batch_size = 100

with tf.Session() as sess:
  summary_writer = tf.summary.FileWriter('experiment',
                                          graph=sess.graph)
  if os.path.isfile("save/model.ckpt"):
    print("Restoring saved parameters")
    saver.restore(sess, "save/model.ckpt")
  else:
    print("Initializing parameters")
    sess.run(tf.global_variables_initializer())

  for step in range(1, n_steps):
    feed_dict = {v: [1], n: [3], p: [5]}
    _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)

    if step % 50 == 0:
      save_path = saver.save(sess, "save/model.ckpt")
      print("Step {0} | Loss: {1}".format(step, cur_loss))

