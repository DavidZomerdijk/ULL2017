# coding: utf-8
from collections import defaultdict
from os import path
import random
import pickle
import numpy as np
import main

#create eval and train set



def read_corpus(path):
    """Read a file as a list of tuples"""

    with open(path, 'r') as f:
        return [tuple(ln.strip().split()) for ln in f]

def createEvalSet(corpus, outputTrain, outputEval, size_eval=1000 ):
    """
    This creates the dataset required for evaluation as described in section 3.1.
    :param corpus: a list of tuples
    :param outputTrain: name of the trainset
    :param outputEval: name of the evaluation set. (tuple of (v,n,v') saved as a pickle file)
    :return:
    """
    random.shuffle(corpus)

    corpus_eval = corpus[:size_eval]
    corpus_train = corpus[size_eval:]

    f_ns_train = defaultdict(int)
    f_vs_train = defaultdict(int)
    f_vn = defaultdict(int)


    # for i in range(len(corpus_train)):
    #     if len(corpus_train[i]) != 2:
    #         print("whoops234!")
    #         print(corpus_train[i])

    for i in range(len(corpus_train)):
        if len(corpus_train[i]) == 2:
            vt,nt = corpus_train[i]
            f_vn[(vt, nt)] += 1
            f_ns_train[nt] += 1
            f_vs_train[vt] += 1

    for vt, nt in corpus_eval:
        f_vn[(vt, nt)] += 1

    evaluation = []
    for i in range(0, len(corpus_eval)):
        # print("whoop2")
        # print(corpus_eval[0])
        # print(f_vs_train[corpus_eval[0]])

        if corpus_eval[i][0] in f_vs_train and corpus_eval[i][1] in f_ns_train:

            conditionsUnsatisfied = True

            while( conditionsUnsatisfied ):

                temp_v = random.choice(list( f_vs_train.keys() ))
                if (temp_v , corpus_eval[i][1] ) not in f_vn:
                    conditionsUnsatisfied = False
                    evaluation.append( (corpus_eval[i][0], corpus_eval[i][1], temp_v)  )

    # pickle.dump( corpus_train, open(outputTrain, "wb"))

    with open(outputTrain, "a") as f:
        f.write("\n".join([ " ".join(element)  for element in corpus_train]) )

    pickle.dump( evaluation, open(outputEval, "wb"))



def create_datasets():
    data_path = path.join(path.dirname(__file__), '..', 'data')
    gold_corpus = path.join(data_path, 'gold_deps.txt')
    all_pairs = path.join(data_path, 'all_pairs')
    corpus_a = read_corpus(all_pairs)

    createEvalSet(corpus_a, "../data/train_eval/all_train_3000.txt", "../data/train_eval/all_eval_3000.p", 3000)

    corpus_g = read_corpus(gold_corpus)
    createEvalSet(corpus_g, "../data/train_eval/gold_corpus_train_1000.txt", "../data/train_eval/gold_corpus_eval_1000.p",
                  1000)

def train_models():
    path_models = "../data/models"
    path_train_data = "../data/train_eval"
    name_train_data = "gold_corpus_train_1000.txt"
    train_data = main.Dataset.load( path.join(path_train_data, name_train_data) )

    #parameters is a tuple of (number of classes, number of iterations)
    parameters = [(15,20),(40,50) ]


    for param in parameters:
        model = main.LSCVerbClasses(train_data, n_cs=param[0], em_iters=param[1])
        model.train()
        outputName = path.join(path_models, (name_train_data[:-4] + "_c=" + str(param[0]) + "i=_" + str(param[1]) + ".p" ))
        pickle.dump(model, open(outputName, "wb"))

    return

def evaluate_models(modelFiles, evaluationFile):
    evaluation = pickle.load(open(evaluationFile, 'rb'))

    for modelfile in modelFiles:
        model = pickle.load(open(modelfile, 'rb'))

        print(model.p_vc)
        count_biggerthen = 0

        # for v, n in evaluation:




    return




if __name__ == "__main__":
    # create_datasets()
    # train_models()

    modelfiles = ["../data/models/gold_corpus_train_1000_c=15i=_20.p", "../data/models/gold_corpus_train_1000_c=40i=_50.p"]
    evaluationFile = "../data/train_eval/gold_corpus_eval_1000.p"
    evaluate_models(modelfiles, evaluationFile)



