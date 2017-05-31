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
img_path = "../../out/img"

iterations = [x * 5 for x in range(31)] #[0,5,10,15,20,25,30,35,40,45,50,45,50]
clusters = [5,10,20,30,40,50,75,100,200,300]
accuracies = dict()
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
plt.savefig(img_path + 'accuracy_clusters.pdf')

#accuracy for several clusters
data =  [(cluster, likelihoods[cluster][50])  for cluster in clusters]
max_iter = 20
x_val = range(0,max_iter)
y_val_5 = [math.log1p( -likelihoods[5][iteration])for iteration in range(0,max_iter)]
y_val_50 = [math.log1p( -likelihoods[50][iteration]) for iteration in range(0,max_iter)]
y_val_100 = [math.log1p( -likelihoods[100][iteration]) for iteration in range(0,max_iter)]
y_val_300 = [math.log1p( -likelihoods[300][iteration]) for iteration in range(0,max_iter)]

line1, = plt.plot(x_val,y_val_5,   label="5 classes" )
line2, = plt.plot(x_val,y_val_50,  label="50 classes" )
line3, = plt.plot(x_val,y_val_100,  label="100 classes" )
line4, = plt.plot(x_val,y_val_300,  label="300 classes" )

plt.legend(handles=[line4,line3,line2,line1], loc=1)
# plt.axis([0, 150, 0.50, 0.82])
plt.xlim([0, max_iter])
# plt.yscale('log')
plt.xlabel('number of iterations')
plt.ylabel('- log likelihood')
plt.savefig('likelihood_part1.pdf')




#likelihood part 2
clusters = [5,50,100,300]
likelihoods = dict()

for cluster in clusters:
    model_path = path + "all_pairs_intransitive_class-" + str(cluster)+"-150.pkl"
    model = pickle.load(open(model_path, 'rb'))
    likelihoods[cluster] = model.likelihoods

x_val = range(1,50)
y_val_5 = [-likelihoods[5][iteration] for iteration in range(1,50)]
y_val_50 = [-likelihoods[50][iteration] for iteration in range(1,50)]
y_val_100 = [-likelihoods[100][iteration] for iteration in range(1,50)]
y_val_300 = [-likelihoods[300][iteration] for iteration in range(1,50)]

line1, = plt.plot(x_val,y_val_5,   label="5 classes" )
line2, = plt.plot(x_val,y_val_50,  label="50 classes" )
line3, = plt.plot(x_val,y_val_100,  label="100 classes" )
line4, = plt.plot(x_val,y_val_300,  label="300 classes" )

plt.legend(handles=[line4,line3,line2,line1], loc=1)

plt.xlabel('number of iterations')
plt.ylabel('- likelihood')
plt.yscale('log')
plt.axis('tight')
plt.savefig(img_path +'likelihood_intransitive.pdf')
