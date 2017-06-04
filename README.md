# Inducing a Semantically Annotated Lexicon via Deep Variational Autoencoders and EM-Based Clustering

An implementation of the paper: [*"Inducing a Semantically Annotated Lexicon via EM-Based Clustering"* by Mats **Rooth**, Stefan Riezler, Detlef Prescher, Glenn Carroll  and Franz Beil (May **1999**)](https://arxiv.org/abs/cs/9905008), which we extended to cluster verb-noun subcategorization frames with a Variational Autoencoder.

The original paper introduced **latent semantic clustering**, an unsupervised method to find good clusters of words from a dependency parsed corpus using an EM Algorithm.

We wrote a report on this implementation as well, which can be found [here](https://www.overleaf.com/read/bmcfwrvyqnkq).

Our implementation is divided up into three steps:

- 1: LSC, Verb classes (corresponding to Section 2 in the original paper), with Evaluation from (Section 3)
- 2: Lexical Induction, Subject/Object classes (corresponding to Section 4 in the original paper)
- 3: VAE reconstruction of the subcategorization frame

This implementation was done as an assignment for the Unsupervised Language Learning course at the University of Amsterdam in 2017 by [Maurits Bleeker](https://github.com/MBleeker), [Thijs Scheepers](http://github.com/tscheepers) and [David Zomerdijk](https://github.com/DavidZomerdijk/).
