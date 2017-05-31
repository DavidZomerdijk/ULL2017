import pickle
import sys
import math
sys.path.append("..")
from lsc_verb_noun_pairs import LSCVerbClasses
import mai
from lsc_subj_intransitive_verbs import SubjectIntransitiveVerbClasses
import lsc_subj_obj_transitive_verbs
import matplotlib.pyplot as plt


path = "../../out/"

iterations = [x * 5 for x in range(31)] #[0,5,10,15,20,25,30,35,40,45,50,45,50]
clusters = [5,10,20,30,40,50,75,100,200,300]
# accuracies = dict()
likelihoods = dict()

for cluster in clusters:
    model_path = path + "all_pairs_lcs-" + str(cluster)+"-150.pkl"
    model = pickle.load(open(model_path, 'rb'))
    accuracies[cluster] = model.accuracies
    likelihoods[cluster] = model.likelihoods

#Accuracy per cluster
n_of_iterations = i
data =  [(cluster, accuracies[cluster][n_of_iterations])  for cluster in clusters]

x_val = [x[0] for x in data]
y_val = [x[1] for x in data]
line1, = plt.plot(x_val,y_val, marker = 'o',label="after " + str(i) + " iterations")
plt.legend(handles=[line1], loc=4)
plt.axis([0, 305, 0.70, 0.82])
plt.xlabel('number of classes')
plt.ylabel('accuracy')
plt.savefig(img_path + 'accuracy_clusters.png')
plt.show()