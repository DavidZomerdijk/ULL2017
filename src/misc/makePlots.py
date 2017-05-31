import pickle
import sys
import math
sys.path.append("..")
from lsc_verb_noun_pairs import LSCVerbClasses
import main
from lsc_subj_intransitive_verbs import SubjectIntransitiveVerbClasses
import lsc_subj_obj_transitive_verbs
import matplotlib.pyplot as plt


path = "../../out/"
img_path = "../../out/img/"

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
n_of_iterations = 150
data =  [(cluster, accuracies[cluster][n_of_iterations])  for cluster in clusters]

x_val = [x[0] for x in data]
y_val = [x[1] for x in data]
line1, = plt.plot(x_val,y_val, marker = 'o',label="after iterations")
plt.legend(handles=[line1], loc=4)
plt.axis([0, 305, 0.70, 0.82])
plt.xlabel('number of classes')
plt.ylabel('accuracy')
plt.savefig(img_path + 'accuracy_clusters.pdf')
plt.close()


#iterations per cluster
x_val = iterations
y_val_5 = [accuracies[5][iteration] for iteration in iterations]
y_val_10 = [accuracies[10][iteration] for iteration in iterations]
y_val_20 = [accuracies[20][iteration] for iteration in iterations]
y_val_30 = [accuracies[20][iteration] for iteration in iterations]
y_val_40 = [accuracies[40][iteration] for iteration in iterations]
y_val_50 = [accuracies[50][iteration] for iteration in iterations]
y_val_75 = [accuracies[75][iteration] for iteration in iterations]
y_val_100 = [accuracies[100][iteration] for iteration in iterations]
y_val_300 = [accuracies[300][iteration] for iteration in iterations]

line1, = plt.plot(x_val,y_val_5,  marker='s', label="5 classes" )
line2, = plt.plot(x_val,y_val_50, marker='o', label="50 classes" )
line3, = plt.plot(x_val,y_val_100, marker='p', label="100 classes" )
line4, = plt.plot(x_val,y_val_300, marker='v', label="300 classes" )

#extra
line5, = plt.plot(x_val,y_val_10, marker='v', label="10 classes" )
line6, = plt.plot(x_val,y_val_20, marker='v', label="20 classes" )
line7, = plt.plot(x_val,y_val_30, marker='v', label="30 classes" )
line8, = plt.plot(x_val,y_val_40, marker='v', label="40 classes" )
line9, = plt.plot(x_val,y_val_75, marker='v', label="75 classes" )

plt.legend(handles=[line1, line5, line6, line2,line3, line4], loc=4)
plt.axis([0, 150, 0.50, 0.90])
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
# plt.axis('tight')
plt.savefig(img_path + 'accuracy_iterations.pdf')
plt.close()


#Likelihood progression
# data =  [(cluster, likelihoods[cluster][50])  for cluster in clusters]
max_iter = 20
x_val = range(1,max_iter)
y_val_5 = [likelihoods[5][iteration]for iteration in range(1,max_iter)]
# y_val_50 = [math.log1p( -likelihoods[50][iteration]) for iteration in range(1,max_iter)]
y_val_50 = [likelihoods[50][iteration] for iteration in range(1,max_iter)]
y_val_100 = [likelihoods[100][iteration] for iteration in range(1,max_iter)]
y_val_300 = [likelihoods[300][iteration] for iteration in range(1,max_iter)]

line1, = plt.plot(x_val,y_val_5,   label="5 classes" )
line2, = plt.plot(x_val,y_val_50,  label="50 classes" )
line3, = plt.plot(x_val,y_val_100,  label="100 classes" )
line4, = plt.plot(x_val,y_val_300,  label="300 classes" )

plt.legend(handles=[line4,line3,line2,line1], loc=1)
plt.xlim([1, max_iter])
plt.xlabel('number of iterations')
plt.ylabel('log likelihood')
plt.savefig(img_path +'likelihood_part1.pdf')
plt.close()

#likelihood part 2

clusters = [5,50,100,300]
likelihoods = dict()

for cluster in clusters:
    model_path = path + "all_pairs_intransitive_class-" + str(cluster)+"-150.pkl"
    model = pickle.load(open(model_path, 'rb'))
    likelihoods[cluster] = model.likelihoods

x_val = range(1,50)
y_val_5 = [likelihoods[5][iteration] for iteration in range(1,50)]
y_val_50 = [likelihoods[50][iteration] for iteration in range(1,50)]
y_val_100 = [likelihoods[100][iteration] for iteration in range(1,50)]
y_val_300 = [likelihoods[300][iteration] for iteration in range(1,50)]

line1, = plt.plot(x_val,y_val_5,   label="5 classes" )
line2, = plt.plot(x_val,y_val_50,  label="50 classes" )
line3, = plt.plot(x_val,y_val_100,  label="100 classes" )
line4, = plt.plot(x_val,y_val_300,  label="300 classes" )

plt.legend(handles=[line4,line3,line2,line1], loc=1)

plt.xlabel('number of iterations')
plt.ylabel('log likelihood')
plt.savefig(img_path +'likelihood_part2.pdf')
plt.close()
