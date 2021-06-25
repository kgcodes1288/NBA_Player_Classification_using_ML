import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scipy.sparse.linalg as ll
import math
from sklearn.decomposition import PCA
import matplotlib
import os
import feather


 #results will be stored here
nba_data = feather.read_dataframe('model_data.feather')



def accuracy(col,results):
    temp = results[col].value_counts().reset_index()
    temp[temp['index'] == True]
    if False not in temp['index'].unique():
        return 100
    else:
        true_count = temp[temp['index'] == True].reset_index(drop=True)[col][0]
        accuracy = (true_count * 100)/len(results)
        return accuracy

    
def plotboundary(labels_mesh,title,imagename,XX,YY,pca_X,y_train):
    plt.contourf(XX,YY,labels_mesh.reshape(XX.shape))
    plt.scatter(pca_X[:,0],pca_X[:,1],c = y_train,cmap = matplotlib.colors.ListedColormap(['red','blue']))
    plt.title(title)
    plt.savefig(imagename + '.jpg')
    




def compareclassifiers(df_data,player):
    results = pd.DataFrame()
    accuracy_dict ={'NB':[],'LOG':[],'knn':[]}
    df_data = df_data.drop(columns=['GAME_ID','PLAYER'])

    X_train, X_test, y_train, y_test = train_test_split(df_data, df_data['CRUNCH'], test_size = 0.2)

    classifier1 = GaussianNB(var_smoothing=1e-03)
    classifier1.fit(X_train, y_train)
    results['NB'] = classifier1.predict(X_test) == y_test

    classifier2 = LogisticRegression(solver='liblinear')
    classifier2.fit(X_train, y_train)
    results['LOG'] = classifier2.predict(X_test) == y_test

    classifier3 = KNeighborsClassifier()
    classifier3.fit(X_train, y_train)
    results['knn'] = classifier3.predict(X_test) == y_test

    for col in ['NB','LOG','knn']:
        acc = accuracy(col,results)
        accuracy_dict[col].append(acc)
        #print("for random_state ",random_state," the accuracy for ", col, "is ",  acc)


    for col in ['NB','LOG','knn']:
        accuracy_dict[col] = sum(accuracy_dict[col])/len(accuracy_dict[col])
        print("Accuracy of the {0} classifier is {1}%".format(col,accuracy_dict[col]))


    #PCA implementation:
    pca = PCA(n_components=2)
    pca.fit(X_train)
    pca_X = pca.fit_transform(X_train)


    NB_class = GaussianNB(var_smoothing=1e-03)
    NB_class.fit(pca_X,y_train)
    LR_class = LogisticRegression(solver='liblinear')
    LR_class.fit(pca_X,y_train)
    KNN_class = KNeighborsClassifier()
    KNN_class.fit(pca_X,y_train)


    a = np.arange(start = pca_X[:,0].min()-0.5, stop = pca_X[:,0].max()+0.5, step = 0.1)
    b = np.arange(start = pca_X[:,1].min()-0.5, stop = pca_X[:,1].max()+0.5, step = 0.1)

    XX,YY = np.meshgrid(a,b)
    mesh_X = np.array([XX.ravel(),YY.ravel()]).T
        
    #plotting for NB
    NB_y_predict_mesh = NB_class.predict(mesh_X)
    plotboundary(NB_y_predict_mesh,'Naive Bayes decision boundary plot_'+player,'NaiveBayes_'+player,XX,YY,pca_X,y_train)

    #plotting for KNN
    knn_y_predict_mesh = KNN_class.predict(mesh_X)
    plotboundary(knn_y_predict_mesh,'KNN decision boundary plot_'+player,'KNN_'+player,XX,YY,pca_X,y_train)

    #plotting for LOG
    log_y_predict_mesh = LR_class.predict(mesh_X)
    plotboundary(log_y_predict_mesh,'LogisticRegression decision boundary plot_'+player,'LogisticRegression_'+player,XX,YY,pca_X,y_train)


compareclassifiers(nba_data,'All')
for player in nba_data['PLAYER'].unique():
    compareclassifiers(nba_data[nba_data['PLAYER']==player],player)
