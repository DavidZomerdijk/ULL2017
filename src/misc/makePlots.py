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

iterations = [x * 5 for x in range(1,21)]  #[0,5,10,15,20,25,30,35,40,45,50,45,50]
clusters = [5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500]
accuracies = dict()
likelihoods = dict()

for cluster in clusters:
    model_path = path + "all_pairs_lcs-" + str(cluster)+"-100.pkl"
    model = pickle.load(open(model_path, 'rb'))
    accuracies[cluster] = model.accuracies
    likelihoods[cluster] = model.likelihoods

#Accuracy per cluster
n_of_iterations = 100
data1 =  [(cluster, accuracies[cluster][10])  for cluster in clusters]
data2 =  [(cluster, accuracies[cluster][50])  for cluster in clusters]
data3 =  [(cluster, accuracies[cluster][75])  for cluster in clusters]
data4 =  [(cluster, accuracies[cluster][100])  for cluster in clusters]

x_val = [x[0] for x in data1]
y_val1 = [x[1] for x in data1]
y_val2 = [x[1] for x in data2]
y_val3 = [x[1] for x in data3]
y_val4 = [x[1] for x in data4]

line1, = plt.plot(x_val,y_val1, marker = 'o',label="after 10 iterations")
line2, = plt.plot(x_val,y_val2, marker = 'v',label="after 50 iterations")
line3, = plt.plot(x_val,y_val3, marker = '^',label="after 75 iterations")
line4, = plt.plot(x_val,y_val4, marker = 'x',label="after 100 iterations")
plt.legend(handles=[line1, line2, line3, line4], loc=4)
plt.axis([0, 505, 0.70, 0.80])
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
y_val_400 = [accuracies[400][iteration] for iteration in iterations]
y_val_500 = [accuracies[500][iteration] for iteration in iterations]

line1, = plt.plot(x_val,y_val_5,  marker='s', label="5 classes" )
line2, = plt.plot(x_val,y_val_50, marker='o', label="50 classes" )
line3, = plt.plot(x_val,y_val_100, marker='p', label="100 classes" )
line4, = plt.plot(x_val,y_val_300, marker='v', label="300 classes" )

#extra
line5, = plt.plot(x_val,y_val_10, marker='v', label="10 classes" )
# line6, = plt.plot(x_val,y_val_20, marker='v', label="20 classes" )
# line7, = plt.plot(x_val,y_val_30, marker='v', label="30 classes" )
# line8, = plt.plot(x_val,y_val_40, marker='v', label="40 classes" )
# line9, = plt.plot(x_val,y_val_75, marker='v', label="75 classes" )
# line10, = plt.plot(x_val,y_val_400, marker='x', label="400 classes" )
line11, = plt.plot(x_val,y_val_500, marker='x', label="500 classes" )

plt.legend(handles=[line1, line5, line2, line3, line4, line11], loc=4)
plt.axis([3, 102, 0.55, 0.80])
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
# plt.axis('tight')
plt.savefig(img_path + 'accuracy_iterations.pdf')
plt.close()


#Likelihood progression
# data =  [(cluster, likelihoods[cluster][50])  for cluster in clusters]
max_iter = 40
x_val = range(1,max_iter)
y_val_5 = [likelihoods[5][iteration]for iteration in range(1,max_iter)]
# y_val_50 = [math.log1p( -likelihoods[50][iteration]) for iteration in range(1,max_iter)]
y_val_50 = [likelihoods[50][iteration] for iteration in range(1,max_iter)]
y_val_100 = [likelihoods[100][iteration] for iteration in range(1,max_iter)]
y_val_300 = [likelihoods[300][iteration] for iteration in range(1,max_iter)]
y_val_500 = [likelihoods[500][iteration] for iteration in range(1,max_iter)]

line1, = plt.plot(x_val,y_val_5,   label="5 classes" )
line2, = plt.plot(x_val,y_val_50,  label="50 classes" )
line3, = plt.plot(x_val,y_val_100,  label="100 classes" )
line4, = plt.plot(x_val,y_val_300,  label="300 classes" )
line5, = plt.plot(x_val,y_val_500,  label="500 classes" )

plt.legend(handles=[line5,line4,line3,line2,line1], loc=4)
plt.xlim([1, max_iter-1])
plt.xlabel('number of iterations')
plt.ylabel('log likelihood')
plt.savefig(img_path +'likelihood_part1.pdf')
plt.close()



#likelihood part 2

clusters = [5,50,100,300,500]
likelihoods = dict()

for cluster in clusters:
    model_path = path + "all_pairs_intransitive_class-" + str(cluster)+"-100.pkl"
    model = pickle.load(open(model_path, 'rb'))
    likelihoods[cluster] = model.likelihoods

max_iter = 15

x_val = range(1,max_iter)
y_val_5 = [likelihoods[5][iteration] for iteration in range(1,max_iter)]
y_val_50 = [likelihoods[50][iteration] for iteration in range(1,max_iter)]
y_val_100 = [likelihoods[100][iteration] for iteration in range(1,max_iter)]
y_val_300 = [likelihoods[300][iteration] for iteration in range(1,max_iter)]
y_val_500 = [likelihoods[500][iteration] for iteration in range(1,max_iter)]

line1, = plt.plot(x_val, y_val_5, label="5 classes")
line2, = plt.plot(x_val, y_val_50, label="50 classes")
line3, = plt.plot(x_val, y_val_100, label="100 classes")
line4, = plt.plot(x_val, y_val_300, label="300 classes")
line5, = plt.plot(x_val, y_val_500, label="500 classes")

plt.legend(handles=[line5, line4, line3,line2,line1], loc=4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([1, max_iter-1])
plt.xlabel('number of iterations')
plt.ylabel('log likelihood')
plt.savefig(img_path + 'likelihood_part2.pdf')
plt.close()

v_accs, n_accs, p_accs, pda_accs, test_losses, train_losses = \
    pickle.load(open(path + '/tf2/plot_results.pkl', 'rb'))

iterations = [x * 10 for x in range(0, 829)]
x_val_train = iterations
y_val_train = [train_losses[iteration]for iteration in iterations]

iterations = [x * 20 for x in range(0, 12)] + [x * 100 for x in range(3, 82)]
x_val_test = iterations
y_val_test = [test_losses[iteration] for iteration in iterations]

line1, = plt.plot(x_val_train, y_val_train, label="train")
line2, = plt.plot(x_val_test, y_val_test, label="test")

plt.legend(handles=[line1, line2], loc=1)

plt.xlim([0, 5000])
plt.ylim([2, 15])
plt.xlabel('number of steps')
plt.ylabel('loss')
plt.savefig(img_path + 'vae_loss.pdf')
plt.close()



iterations = [x * 20 for x in range(0, 12)] + [x * 100 for x in range(3, 82)]
x_val = [0] + iterations
y_val1 = [0] + [v_accs[iteration] for iteration in iterations]
y_val2 = [0] + [n_accs[iteration] for iteration in iterations]
y_val3 = [0] + [p_accs[iteration] for iteration in iterations]

line1, = plt.plot(x_val, y_val1, label="verb accuracy" )
line2, = plt.plot(x_val, y_val2, label="noun accuracy" )
line3, = plt.plot(x_val, y_val3, label="subcategorization frame accuracy" )

plt.legend(handles=[line1, line2, line3], loc=4)

plt.xlim([0, 5000])
plt.xlabel('number of steps')
plt.ylabel('accuracy on the test set')
plt.savefig(img_path + 'vae_acc.pdf')
plt.close()


iterations = [x * 20 for x in range(0, 12)] + [x * 100 for x in range(3, 120)]
x_val = [0] + iterations
y_val1 = [0] + [pda_accs[iteration] for iteration in iterations]

line1, = plt.plot(x_val, y_val1, label="pseudo disambiguation accuracy")

plt.legend(handles=[line1,], loc=1)

plt.xlim([0, 10000])
plt.ylim([0.45, 0.8])
plt.xlabel('number of steps')
plt.ylabel('accuracy on the test set')
plt.savefig(img_path + 'vae_pd_acc.pdf')
plt.close()

line1, = plt.plot(x_val, y_val1, label="pseudo disambiguation accuracy")

plt.legend(handles=[line1,], loc=1)

plt.xlim([0, 250])
plt.ylim([0.45, 0.8])
plt.xlabel('number of steps')
plt.ylabel('accuracy on the test set')
plt.savefig(img_path + 'vae_pd_acc_250.pdf')
plt.close()