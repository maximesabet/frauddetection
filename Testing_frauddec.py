from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from matplotlib import style
style.use("ggplot")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import time

#Directory for indepent variables set and dependent variable set (two different files)

X = np.array(pd.read_csv('C:\Users\Maxime\Documents\Python Books\independent_variables_fraudtests.csv'))
y = np.array(pd.read_csv('C:\Users\Maxime\Documents\Python Books\dependent_variable_fraudec.csv'))
y = y.ravel()
print y
print X
print y.shape
print X.shape
logreg = LogisticRegression()
logreg.fit(X,y)

#logistic regression is normally used if you have a dummy varible 
#train test split; takes a random test size of the percentage mentioned of of the set (0.4), then trains the rest and test. Then it gives you the accuracy as it test the set and gives the coefficients

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

print logreg.coef_ 


#same but with a specified random set (always same results)
    
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

print logreg.coef_  '''

#kcross fit to find the best number of neighbors 
'''
knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn, X, y, cv= 10, scoring = 'accuracy')
print scores
print scores.mean()

k_range = range(31, 51)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    k_scores.append(scores.mean())
print k_scores

plt.plot(k_range, k_scores)
plt.xlabel('Value of k for knn')    
plt.ylabel('Cross-Validated Accuracy')  
plt.show()
'''
#kcross fit with grid (seems a bit less good)     
'''

k_range = range(20, 51)
print k_range
param_grid = dict(n_neighbors = k_range)
print param_grid
grid = GridSearchCV(knn, param_grid, cv = 10, scoring='accuracy')
print grid.fit(X,y)
print grid.grid_scores_
print grid.grid_scores_[0].parameters
print grid.grid_scores_[0].cv_validation_scores
print grid.grid_scores_[0].mean_validation_score
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print grid_mean_scores
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validated Accuracy')
print grid.best_score_
print grid.best_params_
print grid.best_estimator_

pylab.show() '''

#kcross fit with grid and randomized automation; its supposed to be faster but some times wont give you the best results

'''weight_options = ['views']
param_dist = dict(n_neighbors = k_range, weights = weight_options)
rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10, random_state = 5)
print rand.fit(X,y)
print rand.grid_scores
print rand.best_score_
print rand.best_params_'''


#Kmeans technique 
'''
kmeans = MiniBatchKMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)


colors = ["g.","r."]

for i in range(len(X)):
    #print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:, 0], centroids[:, 1], mark = "x", s=150, linewidths = 5, zorder = 10)

plt.show()
'''
#n_clusters=3# Compute clustering with Means
#batch_size= 45


##############################################################################
# Compute clustering with MiniBatchKMeans
'''
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0
mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_
mbk_means_labels_unique = np.unique(mbk_means_labels)


colors = ["g.", "r.", "c."]
for i in range(len(X)):
    plt.plot(X[i][1], X[i][1], colors[mbk_means_labels_unique[i]], makersize = 10)
    
plt.scatter(mbk_means_cluster_centers[:, 0], mbk_means_cluster_centers[:, 1], marker = "x", s=150, linewidths = 5, zorder=10)
    
plt.show() '''