import networkx as nx
import numpy as np
from dbscan import DBSCAN
from random import randint

class mapper:
    def __init__(self, points, filter, cover, epsilon, neighbors, metric) -> None:
        # filtering points using passed filter function. filtered points is an array with same 0 dimension as the points array
        filtered_points = filter(points)
        # getting cover with passed covering function.The covering function must return a cover. A cover is a matrix with one row for each set in the cover and one column for each point in the points array. Each row is a boolean array representing the points which are included in that set
        # a value of 1 in position i,j means that the point at position j in the points array is included in set i of the cover
        cover = cover(filtered_points)

        # empty arrays that the cluster_preimage function will populate
        self.nerve = []
        self.color_map = []

        # mapping
        self.cluster_preimages(points, cover, epsilon, neighbors, metric)
        
    def cluster_preimages(self, points, cover, epsilon, neighbors, metric) -> None:
        # for each row/set in the cover
        for i in cover:
            # get the nonzero indices 
            preimage_indices = np.nonzero(i)[0]
            # get the points corresponding to those indices
            preimage = points[preimage_indices]
            # cluster using my dbscan implementation
            clustered_preimage_indices = DBSCAN(preimage, epsilon, neighbors, metric).clustered_indices
            # assign a color to the current set in the cover, note that all clusters in this preimage will have the same color
            color = '#%06X' % randint(0, 0xFFFFFF)
            for cluster in clustered_preimage_indices:
                # add cluster to nerve 
                self.nerve.append(set(preimage_indices[cluster]))
                # add color of set to color map
                self.color_map.append(color)

        return 
    
    def graph_nerve(self) -> None:
        # make graph object
        nerve_graph = nx.Graph()
        # get iterator for number of vertices in nerve
        nerve_range = range(len(self.nerve))
        # preemptively add all vertices to the graph
        nerve_graph.add_nodes_from([i for i in nerve_range])
        # for each vertex
        for i in nerve_range:
            # for each vertex
            for j in nerve_range:
                # get intersection
                vertex_intersection = self.nerve[i].intersection(self.nerve[j])
                # if intersection is not empty and intersection is not the set itself (this prevents adding simple loops to the graph)
                if len(vertex_intersection) > 0 and len(vertex_intersection) != len(self.nerve[i]):
                    # add edge between the two vertices
                    nerve_graph.add_edge(i, j)
        nx.draw(nerve_graph, with_labels=True, node_color=self.color_map)

        return
