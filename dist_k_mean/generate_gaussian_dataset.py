from math import floor

import numpy as np


def generate_k_gaussians(dimensions=5, num_points=1000000, K=50, alpha=0.0, std_dev=0.1):
    N = np.random.normal(scale=std_dev, size=(num_points, dimensions,))
    centers = np.random.uniform(size=(K, dimensions,))
    the_sum = sum(i ** alpha for i in range(1, K + 1))
    cluster_sizes = [floor(num_points * ((i ** alpha) / the_sum)) for i in range(1, K + 1)]
    N = N[:sum(cluster_sizes)]

    cluster_slices = []
    next_element = 0
    for cluster_size in cluster_sizes:
        cluster_slices.append((slice(next_element, next_element + cluster_size)))
        next_element += cluster_size

    for k, cluster_slice in enumerate(cluster_slices):
        N[cluster_slice] += centers[k]

    return N


if __name__ == '__main__':
    print(generate_k_gaussians())
