import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
from copy import deepcopy
from itertools import combinations
from scipy.spatial.distance import euclidean

def k_means(data, clusters_num, max_iterations=20):

    def distance(x, y, axis=None): 
        return np.linalg.norm(x - y, axis=axis)

    data_size = data.shape[0]

    random_subset_indicies = np.random.randint(low=0, high=data_size, size=clusters_num)
    cluster_centers = data[random_subset_indicies, :]
    cluster_by_item = np.zeros(data_size)
    new_cluster_centers = np.zeros(cluster_centers.shape)
             
    still_change = True
    iteration = 0
    while still_change:
        norms = np.zeros((data_size, clusters_num))
        for cluster in range(clusters_num):
            norms[:, cluster] = distance(data, cluster_centers[cluster], axis=1)
        
        cluster_by_item = np.argmin(norms, axis=1)
        
        for cluster in range(clusters_num):
            cluster_items = [data[item] for item, _cluster in enumerate(cluster_by_item) if cluster == _cluster]
            new_cluster_centers[cluster] = np.mean(cluster_items, axis=0)
            
        still_change = distance(cluster_centers, new_cluster_centers) != 0
        cluster_centers = deepcopy(new_cluster_centers)
        if iteration > max_iterations:
            break
        iteration += 1

    return cluster_centers, cluster_by_item


def DaviesBouldin(data, cluster_centers, cluster_by_item):
    n_clusters = len(set(cluster_by_item))
    unique, counts = np.unique(cluster_by_item, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    s = np.array(
        [sum(euclidean(item, cluster_centers[i]) for item in data[cluster_by_item == i]) / cluster_counts[i]
            for i in range(n_clusters)]
     )

    db = 0

    for i in range(n_clusters):
        max_r = -1
        for j in range(n_clusters):
            if i != j:
                d = euclidean(cluster_centers[i], cluster_centers[j])
                new_r = (s[i] + s[j]) / d
                max_r = new_r if new_r > max_r else max_r
        db += max_r
    return db / n_clusters


def CHIndex(data, cluster_centers, cluster_by_item):
    n_samples, _ = data.shape
    n_clusters = len(set(cluster_by_item))
    unique, counts = np.unique(cluster_by_item, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    gloabal_mean = np.mean(data, axis=0)
 
    b = sum(
        cluster_counts[i] * np.sum((cluster_centers[i] - gloabal_mean) ** 2) for i in range(n_clusters)
    )
 
    w = sum(
        np.sum((data[cluster_by_item == i] - cluster_centers[i]) ** 2)
        for i in range(n_clusters)
    )
    return (n_samples - n_clusters) * b / ((n_clusters - 1) * w)


def RandIndex(true, predicted):
    n = len(true)
    A = B = C = D = 0
 
    for i, j in combinations(range(n), r=2):
        A += 1 if true[i] == true[j] and predicted[i] == predicted[j] else 0
        B += 1 if true[i] != true[j] and predicted[i] != predicted[j] else 0
        C += 1 if true[i] != true[j] and predicted[i] == predicted[j] else 0
        D += 1 if true[i] == true[j] and predicted[i] != predicted[j] else 0
    return (A + B) / (A + B + C + D)


def FWIndex(true, predicted):
    n = len(true)
    TP = FP = TN = FN = 0
 
    for i, j in combinations(range(n), r=2):
        TP += 1 if true[i] == true[j] and predicted[i] == predicted[j] else 0
        FP += 1 if true[i] == true[j] and predicted[i] != predicted[j] else 0
        FN += 1 if true[i] != true[j] and predicted[i] == predicted[j] else 0
        TN += 1 if true[i] != true[j] and predicted[i] != predicted[j] else 0
    return np.sqrt((TP / (TP + FP)) * (TP / (TP + FN)))



def main():
    INPUT_DIR = '/homework/stochastic-course/task4/data' 
    #INPUT_DIR = 'data' 
    
    RESULTS_DIR = '/homework/stochastic-course/task4/results'
    #RESULTS_DIR = 'results'
    
    CLUSTERS_RANGE = range(2, 10)
    IMG_URL = os.path.join(INPUT_DIR, 'policemen.jpg')
    DATA_URL = os.path.join(INPUT_DIR, 'input.txt')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    DB = dict() # Davies-Bouldin
    CH = dict() # Calinski-Harabasz 
    RI = dict() # Rand index 
    FW = dict() # Fowlkes-Mallows

    x_axes = list(CLUSTERS_RANGE)

    print('RUNNING...')
    

    image = np.array(Image.open(IMG_URL))
    data = image.reshape(image.shape[0] * image.shape[1], image.shape[2])  

    print("Evaluating of Davies-Bouldin and Calinski-Harabasz")
    for k in CLUSTERS_RANGE:
        if k < 2: continue
        cluster_centers, cluster_by_item = k_means(data, k)
        DB[k] = DaviesBouldin(data, cluster_centers, cluster_by_item) # Davies-Bouldin
        CH[k] = CHIndex(data, cluster_centers, cluster_by_item) # Calinski-Harabasz
        

    print("Saving plots...")
    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(DB.values()))
    ax.set_title("Davies-Bouldin")
    plt.savefig(os.path.join(RESULTS_DIR, "Davies_Bouldin.png"))

    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(CH.values()))
    ax.set_title("Calinski-Harabasz")
    plt.savefig(os.path.join(RESULTS_DIR, "Calinski_Harabasz.png"))


    # Save policeman
    best_score = int(input("clusters> "))
    print("Saving the image...")
    centroids, clusters = k_means(data, best_score)
    compressed_img_url = os.path.join(RESULTS_DIR, '{}_clusters_policemen.jpg'.format(best_score)) 
    compressed_img = np.vstack([centroids[i] for i in clusters]).astype(np.uint8).reshape(image.shape)
    Image.fromarray(compressed_img).save(compressed_img_url)
    print("Successfully saved!")


    data = np.loadtxt(DATA_URL, delimiter=' ')
    labels, points = data[:, 0], data[:, 1:]
     
    print("Evaluating of Rand index and Fowlkes-Mallows")
    for k in CLUSTERS_RANGE:
        if k < 2: continue
        _, cluster_by_item = k_means(points, k)
        RI[k] = RandIndex(labels, cluster_by_item) # Rand index
        FW[k] = FWIndex(labels, cluster_by_item) # Fowlkes-Mallows


    print("Saving plots...")
    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(RI.values()))
    ax.set_title("Rand index")
    plt.savefig(os.path.join(RESULTS_DIR, "Rand_Index.png"))\

    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(FW.values()))
    ax.set_title("Fowlkes-Mallows")
    plt.savefig(os.path.join(RESULTS_DIR, "Fowlkes_Mallows.png"))

    print('DONE!')


if __name__ == '__main__':
    main()