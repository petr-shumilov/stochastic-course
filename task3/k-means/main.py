import numpy as np
from copy import deepcopy
import os

from PIL import Image

def k_means(data, clusters_num, max_iteration):

    def distance(x, y, axis=None): 
        return np.linalg.norm(x - y, axis=axis)

    data_size = data.shape[0]

    random_subset_indicies = np.random.randint(low=0, high=data_size, size=clusters_num)
    cluster_centers = data[random_subset_indicies, :]
    cluster_by_item = np.zeros(data_size)
    new_cluster_centers = np.zeros((clusters_num, 3))
             
    still_changed = True
    iteration = 0
    while iteration < max_iteration and still_changed:
        norms = np.zeros((data_size, clusters_num))
        for cluster in range(clusters_num):
            norms[:, cluster] = distance(data, cluster_centers[cluster], axis=1)
        
        cluster_by_item = np.argmin(norms, axis=1)
        
        for cluster in range(clusters_num):
            cluster_items = [data[item] for item, _cluster in enumerate(cluster_by_item) if cluster == _cluster]
            if cluster_items:
                new_cluster_centers[cluster] = np.mean(cluster_items, axis=0)

        still_changed = distance(cluster_centers, new_cluster_centers) != 0
        cluster_centers = new_cluster_centers
        iteration += 1
        
    return cluster_centers, cluster_by_item


def main():
    #INPUT_DIR = '/homework/stochastic-course/task3/data' 
    INPUT_DIR = 'data' 
    
    #RESULT_DIR = '/homework/stochastic-course/task3/results'
    RESULT_DIR = 'results'

    NUM_OF_CLUSTERS = [2, 3, 5, 10, 13, 17, 21]
    NUM_OF_ITERATIONS = 5
    

    os.makedirs(RESULT_DIR, exist_ok=True)
    files = [(os.path.splitext(filepath)[0], os.path.splitext(filepath)[1][1:]) for filepath in os.listdir(INPUT_DIR)]
    
    print('RUNNING...')
    for filename, fileext in files:
        
        origin_img_url = os.path.join(INPUT_DIR, '{}.{}'.format(filename,fileext))
        origin_img = np.array(Image.open(origin_img_url), dtype=np.uint8)

        points = origin_img.reshape((origin_img.shape[0] * origin_img.shape[1], origin_img.shape[2]))
        
        for num_clusters in NUM_OF_CLUSTERS:
            
            cluster_centers, cluster_by_item = k_means(points, num_clusters, NUM_OF_ITERATIONS)
            
            compressed_img_url = os.path.join(RESULT_DIR, '{}_splitted_on_{}.{}'.format(filename, num_clusters, fileext))
            compressed_img = np.vstack([cluster_centers[i] for i in cluster_by_item]).astype(np.uint8).reshape(origin_img.shape)
            
            Image.fromarray(compressed_img).save(compressed_img_url)

    print('DONE!')

if __name__ == '__main__':
    main()

