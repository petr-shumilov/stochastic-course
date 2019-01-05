import numpy as np
from copy import deepcopy
import os

from PIL import Image


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


def main():
    INPUT_DIR = 'data' 
    RESULT_DIR = 'results'

    NUM_OF_CLUSTERS = [3, 5, 8]
    NUM_OF_ITERATIONS = 300
    NUM_OF_ATTEMPTS = 3


    os.makedirs(RESULT_DIR, exist_ok=True)
    files = [(os.path.splitext(filepath)[0], os.path.splitext(filepath)[1][1:]) for filepath in os.listdir(INPUT_DIR)]
    
    print('RUNNING...')
    for filename, fileext in files:
        
        origin_img_url = os.path.join(INPUT_DIR, '{}.{}'.format(filename,fileext))
        origin_img = np.array(Image.open(origin_img_url), dtype=np.uint8)

        points = origin_img.reshape((origin_img.shape[0] * origin_img.shape[1], origin_img.shape[2]))
        
        for num_clusters in NUM_OF_CLUSTERS:
            for attempt in range(NUM_OF_ATTEMPTS):

                centroids, clusters = k_means(points, num_clusters, NUM_OF_ITERATIONS)
                
                compressed_img_url = os.path.join(RESULT_DIR, '{}_splitted_on_{}_attempt_{}.{}'.format(filename, num_clusters, attempt, fileext))
                compressed_img = np.vstack([centroids[i] for i in clusters]).astype(np.uint8).reshape(origin_img.shape)
                
                Image.fromarray(compressed_img).save(compressed_img_url)

    print('DONE!')

if __name__ == '__main__':
    main()

