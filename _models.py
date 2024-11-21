import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from helpers import draw_statistic_graph as dsg
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

class SupervisedLearning():

    def Linear_Regression(self, X_train, y_train):
        return LinearRegression().fit(X_train, y_train)

    def Logistic_Regression(self, X_train, y_train):
        return LogisticRegression().fit(X_train, y_train)        

    def Decision_Tree(self, X_train, y_train):
        return DecisionTreeClassifier().fit(X_train, y_train)

    def RandomForest(self, X_train, y_train):
        return RandomForestClassifier().fit(X_train, y_train)
    
    def SVM(self, X_train, y_train):
        return svm.SVC(decision_function_shape='ovo').fit(X_train, y_train)

    def KNN(self, X_train, y_train, n_neighbors=4):
        return KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)

    def Naive_Bayes(self, X_train, y_train):
        return GaussianNB().fit(X_train, y_train)


class UnsupervisedLearning():
    class Dimensionality_Reduction():
        def PCA(self, data, target, n_components=2):

            pca = PCA(n_components=n_components)  
            result_data = pca.fit_transform(data)

            dsg.Explained_Variance_Ratio(pca.explained_variance_ratio_)
            dsg.Dimensionality_Reduction_Visualization(result_data, target)

            return result_data
        
        def TSNE(self, data, target, n_components=2):

            tsne = TSNE(n_components=n_components)
            result_data = tsne.fit_transform(data)

            dsg.Dimensionality_Reduction_Visualization(result_data, target)

            return result_data;
            

    class Clustering():
        def KMeans(self, data, k=4):

            kmeans = KMeans(n_clusters=k, random_state=42).fit(data)

            print(
                    "Inertia:", kmeans.inertia_,
                    "\nLabels:", kmeans.labels_,
                    '\nSilhouette Score:', silhouette_score(data, kmeans.labels_),
                    "\nCentroids:", kmeans.cluster_centers_ 
                )

            return kmeans 

        def Hierarchy_Cluster(self, data, k=4):
            
            prediction = AgglomerativeClustering(n_clusters=k, metric ='euclidean', linkage='average').fit_predict(data)
            
            print(
                    "Clusters:", np.unique(prediction),
                    "Labels:", prediction,
                    '\nSilhouette Score:', silhouette_score(data, prediction),
                )

        def DBSCAN(self, data, radius=1.5, core_points=5):

            prediction = DBSCAN(eps=radius, min_samples=core_points).fit_predict(data) 
            
            print(
                    "Clusters:", np.unique(prediction),
                    "\nLabels:", prediction,
                    '\nSilhouette Score:', silhouette_score(data, prediction),
                )

