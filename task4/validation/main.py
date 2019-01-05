import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
import os
from copy import deepcopy


def k_means(points, k, iterations):

    n = points.shape[0]
    centroids = points[np.random.randint(n, size=k),:]

    new_centroids   = np.zeros(centroids.shape)
    clusters        = np.zeros(n)
    distances       = np.zeros((n, k))

    changed = True
    while changed and iterations > 0:
        for i in range(k):
            distances[:, i] = np.linalg.norm(points - centroids[i], axis=1)

        clusters = np.argmin(distances, axis=1)

        for i in range(k):
            new_centroids[i] = np.mean(points[clusters == i], axis=0)

        changed = np.linalg.norm(centroids - new_centroids) != 0

        centroids = deepcopy(new_centroids)
        iterations -= 1

    return centroids, clusters

def DaviesBouldin(X, centroids, clusters):
    n_cluster = len(centroids)
    cluster_k = [X[clusters == k] for k in range(n_cluster)]
    variances = [np.mean([np.linalg.norm(p - centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []

    for i in range(n_cluster):
        for j in range(n_cluster):
            if i != j:
                db.append((variances[i] + variances[j]) / np.linalg.norm(centroids[i] - centroids[j]))

    return np.max(db) / n_cluster


def CHIndex(X, centroids, clusters):
    k = len(centroids)
    mean = np.mean(centroids, axis=0)
    W = np.sum([(x - centroids[clusters[i]])**2 for i, x in enumerate(X)])
    B = np.sum([len(X[clusters == i]) * (c - mean)**2 for i, c in enumerate(centroids)])
    n = len(X)
    return (n - k) * B / ((k - 1) * W)


def inner_criterias(X, k_range, iterations=300):
    DB = dict()  # Davies-Bouldin
    CH = dict()  # Calinski-Harabasz

    for k in k_range:
        if k < 2:
            continue
        centroids, clusters = k_means(X, k, iterations)
        DB[k] = DaviesBouldin(X, centroids, clusters)
        CH[k] = CHIndex(X, centroids, clusters)

    ch_list = list(CH.items())

    db_best = max(DB, key=DB.get)
    ch_best = 0
    delta = sys.maxsize
    for k in range(1, len(CH) - 1):
        temp = ch_list[k + 1][1] - 2 * ch_list[k][1] + ch_list[k - 1][1]
        if temp < delta:
            delta = temp
            ch_best = ch_list[k][0]

    return DB, db_best, CH, ch_best


def outer_criterias(X, k_range, reference, iterations=300):
    n = len(X)
    RS = dict()  # Rand Statistic
    FM = dict()  # Fowlkes-Mallows

    for k in k_range:
        TP, FN, FP, TN = (0,)*4
        if k < 2:
            continue

        _, clusters = k_means(X, k, iterations)
        # Compute TP, FN, FP, TN.
        for i in range(n):
            for j in range(i + 1, n):
                if clusters[i] == clusters[j] and reference[i] == reference[j]:
                    TP += 1
                elif clusters[i] != clusters[j] and reference[i] == reference[j]:
                    FN += 1
                elif clusters[i] == clusters[j] and reference[i] != reference[j]:
                    FP += 1
                else:
                    TN += 1
        RS[k] = (TP + TN) / n
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        FM[k] = np.sqrt(precision * recall)

    best_rs = max(RS, key=RS.get)
    best_fm = max(FM, key=FM.get)

    return RS, best_rs, FM, best_fm

def main():
    INPUT_DIR = 'data' 
    RESULTS_DIR = 'results'
    CLUSTERS_RANGE = range(2, 10)
    IMG_URL = os.path.join(INPUT_DIR, 'policemen.jpg')
    DATA_URL = os.path.join(INPUT_DIR, 'input.txt')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    

    print('RUNNING...')
    

    # Inner criterias' block.
    image = np.array(Image.open(IMG_URL), dtype=np.uint8)
    new_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    db, best_db, ch, best_ch = inner_criterias(new_image, CLUSTERS_RANGE)
    best_inner = (best_db + best_ch) // 2


    # Save the clustered image.
    centroids, clusters = k_means(new_image, best_inner, iterations=300)
    compressed_img_url = os.path.join(RESULTS_DIR, '{}_clusters_policemen.jpg'.format(best_inner)) 
    compressed_img = np.vstack([centroids[i] for i in clusters]).astype(np.uint8).reshape(image.shape)
    Image.fromarray(compressed_img).save(compressed_img_url)


    # Outer criterias' block.
    data = np.loadtxt(DATA_URL, delimiter=' ')
    reference, points = data[:, 0], data[:, 1:]
    rs, best_rs, fm, best_fm = outer_criterias(points, CLUSTERS_RANGE, reference)

    
    x_axes = list(CLUSTERS_RANGE)

    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(db.values()))
    ax.set_title('Davies-Bouldin. Optimal $k$ is %d' % best_db)
    plt.savefig(os.path.join(RESULTS_DIR, "Davies_Bouldin.png"))

    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(ch.values()))
    ax.set_title('Calinski-Harabasz. Optimal $k$ is %d' % best_ch)
    plt.savefig(os.path.join(RESULTS_DIR, "Calinski_Harabasz.png"))

    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(rs.values()))
    ax.set_title('Rand Statistic. Optimal $k$ is %d' % best_rs)
    plt.savefig(os.path.join(RESULTS_DIR, "Rand_Statistic.png"))\

    _, ax = plt.subplots(1, 1)
    ax.scatter(x=x_axes, y=list(fm.values()))
    ax.set_title('Fowlkes-Mallows. Optimal $k$ is %d' % best_fm)
    plt.savefig(os.path.join(RESULTS_DIR, "Fowlkes_Mallows.png"))

    print('DONE!')

if __name__ == '__main__':
    main()