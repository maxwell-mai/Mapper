import numpy as np
from random import randint
from utils import get_distances, get_adjacency

class DBSCAN:
    def __init__(self, points, epsilon, neighbors, metric, colors=False) -> None:
        self.epsilon = epsilon
        self.neighbors = neighbors
        self.metric = metric

        # clustering
        self.clustered_indices = self.collect_clusters(points)
        # getting an enumerated color map for each cluster with same ordering and shape as initial points list
        if colors:
            self.color_map = self.get_color_map(points)

        # counting clusters
        self.cluster_count = len(self.clustered_indices)

    def get_adjacency(self, points):
        # this method is just meant to tie the get_adjacency function from utils to the dbscan class for convenience
        adjacency = get_adjacency(points, self.metric, self.epsilon, self.neighbors)

        return adjacency

    def collect_clusters(self, points):
        # cluster collection array
        clusters = []

        # tracking array for visited vertices
        seen_points = [False for i in points]

        # sub function to perform breadth first search. Note that this is a part of the collect clusters method because they must both access the seen_points array
        def breadth_first_search(index):
            # adding our starting point to the cluster and queue
            cluster = [index]
            queue = [index]
            # while nodes left to explore
            while queue:
                # get the next node in the queue
                index = queue.pop(0)
                # mark it as visited
                seen_points[index] = True

                # get neighboring points represented by row indices of the adjacency matrix
                distances = get_distances(points, points[index], self.metric)
                neighbor_indices = np.nonzero((distances > 0) & (distances < self.epsilon))[0]

                # for each neighbor
                for i in neighbor_indices:
                    # if we havent seen it yet add it to the queue (so we explore its neighbors), mark it as visited, and add to the cluster
                    if not seen_points[i]:
                        queue.append(i)
                        seen_points[i] = True
                        cluster.append(i)
            # return the cluster
            return cluster
        
        # for each unseen point get the cluster it belongs to. Note that the changes made to seen_points during the call of the breadth first search function are relevant here 
        for i in range(len(points)):
            if not seen_points[i]:
                clusters.append(breadth_first_search(i))
                
        return clusters
    
    def get_color_map(self, points):
        # this function computes a mask array of color hexcodes for each cluster. The returned array can be passed as the color param to a matplotlib scatterplot

        # initialize 0 array of same shape as points array
        color_map = [0 for i in points]

        # iterate through the clusters
        for cluster in self.clustered_indices:
            # select a random color for the cluster
            color = '#%06X' % randint(0, 0xFFFFFF)
            # for each point in the cluster
            for j in cluster:
                # label that point with the clusters color
                color_map[j] = color
        
        return color_map