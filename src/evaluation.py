# coding: utf-8
from collections import defaultdict
from os import path
import random
import pickle
import numpy as np

from main import LSCVerbClasses, Dataset



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
    minimum_count = 30
    maximum_count = 3000
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

    count_v = 0
    count_v_prime = 0

    #create a dictionary only with keys that have the minimum requirements
    f_vs_train_min = dict()
    for key, value in f_vs_train.items():
        if value >= minimum_count:
            f_vs_train_min[key] = value

    evaluation = []
    for i in range(0, len(corpus_eval)):

        # print("whoop2")
        # print(corpus_eval[0])
        # print(f_vs_train[corpus_eval[0]])

        if corpus_eval[i][0] in f_vs_train and corpus_eval[i][1] in f_ns_train and len(corpus_eval[i])==2:
            if f_vs_train[ corpus_eval[i][0] ] >= minimum_count and  f_ns_train[corpus_eval[i][1]] >= minimum_count and f_vs_train[ corpus_eval[i][0] ] <= maximum_count  and f_ns_train[corpus_eval[i][1]] <= maximum_count:
                conditionsUnsatisfied = True

                while( conditionsUnsatisfied ):
                    temp_v = random.choice(list( f_vs_train_min.keys() ))

                    if (temp_v , corpus_eval[i][1] ) not in f_vn and f_vs_train_min[temp_v] <= maximum_count :
                        conditionsUnsatisfied = False
                        evaluation.append( (corpus_eval[i][0] , corpus_eval[i][1], temp_v)  )

                        count_v_prime  += f_vs_train[temp_v]
                        count_v += f_vs_train[corpus_eval[i][0] ]

    # print(len(evaluation))
    # print(count_v, count_v_prime)


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

def train_models(parameters, name_train_data , name):
    path_models = "../data/models"
    path_train_data = "../data/train_eval"


    train_data = Dataset.load( path.join(path_train_data, name_train_data) )


    #parameters is a tuple of (number of classes, number of iterations)
    # parameters = [(1,50), (10,50), (20,50), (30,50), (40,50) , (50,50) , (60,50) , (70,50) , (80,50) , (90,50), (100,50) ]


    for param in parameters:
        model = LSCVerbClasses(train_data, n_cs=param[0], em_iters=param[1],name=name)
        model.train()
        # outputName = path.join(path_models, (name_train_data[:-4] + "_c=" + str(param[0]) + "i=_" + str(param[1]) + ".p" ))
        # pickle.dump(model, open(outputName, "wb"))

    return

def evaluate_models(modelFiles, name_train_data,  evaluationFile):
    evaluation = pickle.load(open(evaluationFile, 'rb'))
    path_models = "../data/models"
    path_train_data = "../data/train_eval"
    # name_train_data = "gold_corpus_train_1000.txt"

    train_data = Dataset.load(path.join(path_train_data, name_train_data))

    with open("evaluation31.txt", "w") as f:
        f.write("")
    print("created evaluation31.txt")


    idx_vs = dict()
    idx_ns = dict()
    # print(train_data.ns)
    for i in range( len(train_data.ns) ):
        idx_ns[ train_data.ns[i] ] = i
    for i in range( len(train_data.vs) ):
        idx_vs[ train_data.vs[i] ] = i

    for clusters, iters, modelfile in modelFiles:
        model = pickle.load(open(modelfile, 'rb'))
        count_biggerthen = 0
        total = 0

        for v, n, v_prime in evaluation:

            # print(v,n,v_prime)
            idx_n = idx_ns[n]
            idx_v = idx_vs[v]
            idx_v_prime  = idx_vs[v_prime]

            # print(idx_n)
            # print(idx_v)


            numerator =       sum( model.p_c ) + np.sum(model.p_vc[idx_v,:])         + sum( model.p_nc[idx_n ,:])
            numerator_prime = sum( model.p_c ) + np.sum(model.p_vc[idx_v_prime,:])   + sum( model.p_nc[idx_n ,:])

            # print(model.p_vc.shape[0])
            # print(np.sum(model.p_vc[idx_v]),      (np.sum(model.p_vc[idx_v_prime])))

            denominator_prime = sum(model.p_c)  * model.p_nc.shape[0] + np.sum(model.p_vc[idx_v_prime,:]) * model.p_nc.shape[0] + np.sum( model.p_nc )
            denominator =       sum(model.p_c ) * model.p_nc.shape[0] + np.sum(model.p_vc[idx_v,:] )      * model.p_nc.shape[0] + np.sum( model.p_nc )
            # print(np.sum(model.p_vc[idx_v_prime]), np.sum(model.p_vc[idx_v] ))

            # print("pc: ",sum( model.p_c) )
            # print("shape p_vc: " ,model.p_vc.shape)
            # print("p_vc: ", sum(model.p_vc[idx_v,:]))
            # print("p_nc", sum( model.p_nc[idx_n, : ]))
            # print(" numerator: ", numerator)
            # print("denominator", denominator)
            # print("sum( model.p_nc ): ",  model.p_nc.shape )
            # self.p_c, self.p_vc, self.p_nc

            P_n_given_v       = numerator / denominator
            P_n_given_v_prime = numerator_prime / denominator_prime
            # if np.sum(model.p_vc[idx_v]) >   np.sum(model.p_vc[idx_v_prime] ):
            #     count_biggerthen +=1


            #
            if P_n_given_v > P_n_given_v_prime:
                count_biggerthen += 1

            #
            # if numerator > numerator_prime:
            #     count_biggerthen += 1

            total += 1

            # print(count_biggerthen, total)


        #
        # print(len(evaluation))
        print(str(clusters) + " " + str(iters))
        print( "Accuracy: " + str(count_biggerthen * 1.0 / total))
        with open("evaluation31.txt", "a") as f:
            f.write( str(clusters) + " " + str(iters) + " " +  str(count_biggerthen * 1.0 / total) + "\n" )



        # for v, n in evaluation:



if __name__ == "__main__":
    # create_datasets()


    # modelfiles = [(15,10,"../out/gold_deps-15-10.pkl"), (40,30, "../out/gold_deps-40-30.pkl"), (40,40,"../out/gold_deps-40-40.pkl")]
    # modelfiles = [ "../out/gold_deps-40-40.pkl"]
    # name_train_data = "gold_corpus_train_1000.txt"
    evaluationFile = "../data/train_eval/gold_corpus_eval_1000.p"

    parameters = [(1, 50), (10, 50), (20, 50), (30, 50), (40, 50), (50, 50), (60, 50), (70, 50), (80, 50), (90, 50),
                  (100, 50)]
    name_train_data = "all_train_3000.txt"
    train_models(parameters, "all_train_3000.txt", "all")

    print("######    trained models     ##########")
    modelfiles = []
    for c, i in parameters:
        for iter in [0,10,20,30,40]:
            modelfiles.append((c, iter, "../out/all-" + str(c) + "-" + str(iter) +".pkl"))

    evaluationFile = "../data/train_eval/all_eval_3000.p"

    evaluate_models(modelfiles, name_train_data, evaluationFile)



