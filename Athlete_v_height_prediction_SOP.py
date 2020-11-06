###Python script to apply various machine learning methods to distinguish athletes from the average person using their height, weight, age and sex
###Downloaded original data from: https://www.theguardian.com/sport/datablog/2012/aug/07/olympics-2012-athletes-age-weight-height#data
###'Average person' data simulated in Excel (e.g. =NORMINV(RAND(), 80,3) for male weight) - completely made up
###Before staring check that you have all these installed:
#Python
import sys
print('Python: {}'.format(sys.version))
#Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55)
#[GCC 7.2.0] on linux2

# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
#scipy: 1.1.0

#numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
#numpy: 1.14.3

#matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
#matplotlib: 2.2.2

#pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
#pandas: 0.23.0

#scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
#sklearn: 0.19.1

##Start here if all above are available on server
#use help(“FunctionName”) if stuck
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets, metrics

##Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pandas.read_csv(url, names=names)
#print(dataset.groupby('class').size())
dataset=pandas.read_table("~/athlete_v_average.txt",sep='\t',header=0)
print(dataset.shape)
#(505,5)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('Athlete').size())

#Open XMing or an X11 display server before proceeding
#Boxplot
dataset.plot(kind='box', subplots=True, layout=(1,5), sharex=False, sharey=False) #Google 'pandas.DataFrame.plot' for details
plt.show()
#Histogram
dataset.hist()
plt.show()
#Scatter plot
scatter_matrix(dataset)
plt.show()

##Split-out validation dataset
array = dataset.values
X = array[:,0:4] #all rows, 1st-3rd column
Y = array[:,4] #all rows, 4th column (i.e. height)
#print(Y)
validation_size = 0.20 #We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset
seed = 11
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

##Test options and evaluation metric
seed = 5
scoring = 'accuracy' #This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate)

##Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression())) #Linear
models.append(('LDA', LinearDiscriminantAnalysis())) #Linear
models.append(('KNN', KNeighborsClassifier())) #non-linear
models.append(('CART', DecisionTreeClassifier())) #non-linear
models.append(('NB', GaussianNB())) #non-linear
models.append(('SVM-Linear', SVC(kernel='linear'))) #by default: 'rbf' (i.e. radial basis function; non-linear) - can take a very long time for large datasets
#evaluate each model in turn
results = []
names = []
kfold = model_selection.KFold(n_splits=10, random_state=seed) #10-fold cross-validation
for name, model in models:
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) #format of results: string, float, float in brackets
	print(msg)

#Results may vary slightly due to different validation set:
#LR: 0.933333 (0.133333)
#LDA: 0.933333 (0.133333)
#KNN: 0.966667 (0.100000)
#CART: 0.966667 (0.100000)
#NB: 0.966667 (0.100000)
#SVM-Linear: 0.933333 (0.133333)

##Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

##Make predictions on validation dataset (Logistic regression)
lr=LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions)) #Recall = TP/TP+FN; Precision = TP/TP+FP
print(confusion_matrix(Y_validation, predictions)) #confusion matrix provides an indication of the errors made
print(classification_report(Y_validation, predictions))

#Download pylib files (i.e. setup.py, plot.py) from https://github.com/PanWu/pylib
#python setup.py develop (--help)
execfile("plot.py")
plot_decision_boundary(lr, X=X_train, Y=Y_train)
plt.show()

##Deep Learning - Logistic regression classifier using Restricted Boltzmann Machines feature extractor
model = BernoulliRBM()
model.fit(X_train) #Unsupervised learning
classifier = Pipeline(steps=[('rbm', model), ('logistic', LogisticRegression(C=100.0))])
classifier.fit(X_train, Y_train)
cv_results = model_selection.cross_val_score(classifier, X_train, Y_train, cv=kfold, scoring=scoring)
results.append(cv_results)
name="Logistic regression using RBM features"
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_train,
        classifier.predict(X_train))))