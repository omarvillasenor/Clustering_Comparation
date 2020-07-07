import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture, metrics
from sklearn.preprocessing import StandardScaler

#Create proof data
np.random.seed(0)
n_samples = 1500
X = 6*[None]

#circulos concéntricos
X[0] = StandardScaler().fit_transform(
    datasets.make_circles(n_samples=n_samples, factor=0.5, noise=.05)[0]
)
#Moons
X[1] = StandardScaler().fit_transform(
    datasets.make_moons(n_samples=n_samples, noise=0.05)[0]
)
#Blobs
X[2] = StandardScaler().fit_transform(
    datasets.make_blobs(n_samples=n_samples, random_state=8)[0]
)
#No clustering plane
X[3] = StandardScaler().fit_transform(
    np.random.rand(n_samples, 2)
)
#Blobs with asintropic deformation
xtemp, _ = datasets.make_blobs(n_samples=n_samples, random_state=170)
X[4] = StandardScaler().fit_transform(
    np.dot(xtemp, [[0.6, -0.6], [-0.4, 0.8]])
)
#Blobs with different variance
X[5] = StandardScaler().fit_transform(
    datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=142)[0]
)

#Number of classes
classes = [2,2,3,3,3,3]

#Draw Function
def model_plotter(X, subtitle="", y=0):
    plt.figure(figsize=(27,9))
    plt.suptitle(subtitle, fontsize=30)
    for i in range(6):
        ax = plt.subplot(2,3,i+1)
        if y is not 0:
            ax.scatter(X[i][:,0],X[i][:,1], c=y[i])
        else:
            ax.scatter(X[i][:,0],X[i][:,1])
    plt.show()

#Models Function
def create_and_run(m, X, classes, args=[], values=[], eps=[]):
    y = []
    ind = 0
    for c, x in zip(classes, X):
        #create model
        model = m()

        #Check Clusters, components or EPS
        if hasattr(model, "n_clusters"): setattr(model, "n_clusters", c)
        elif hasattr(model, "n_components"): setattr(model, "n_components", c)
        if hasattr(model, "eps") and len(eps) > 0: 
            setattr(model, "eps", eps[ind])
            ind+=1
        
        #Check the rest of parameters
        for index, arg in enumerate(args):
            setattr(model, arg, values[index])
        
        #Train
        model.fit(x)
        
        #Save results
        if hasattr(model, 'labels_'):
            y.append(model.labels_.astype(np.int))
        else:
            y.append(model.predict(x))
    
    return y



#Draw data
model_plotter(X, 'Data')

kmeans = create_and_run(cluster.KMeans, X, classes)
model_plotter(X, "KMeans", kmeans)


birch = create_and_run(cluster.Birch, X, classes)
model_plotter(X, "Birch", birch)

spectral_clustering = create_and_run(
    cluster.SpectralClustering,
    X, classes,
    args=["affinity"],
    values=["nearest_neighbors"])
model_plotter(X, "SpectralClustering", spectral_clustering)

gaussian_mixture = create_and_run(
    mixture.GaussianMixture, 
    X, 
    classes, 
    args=["covariance_type"],
    values=["full"])
model_plotter(X, "GaussianMixture", gaussian_mixture)


optics = create_and_run(
    cluster.OPTICS, 
    X, 
    classes, 
    args=["min_samples", "xi", "min_cluster_size"],
    values=[20, 0.05, 0.1])
model_plotter(X, "OPTICS", optics)


dbscan = create_and_run(
    cluster.DBSCAN, 
    X, 
    classes,
    eps=[0.3,0.3,0.3,0.3,0.15, 0.18])
model_plotter(X, "DBSCAN", dbscan)
