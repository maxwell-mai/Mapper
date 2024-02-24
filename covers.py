import numpy as np 
from dbscan import DBSCAN

'''
I represent covers as an m x n matrix where n is the number of input points (i.e. the number of outputs from my filter function), and m is the number of sets in my cover. 
Each row represents a single set of the cover. 
The entries are boolean with a 1 in entry i, j meaning that the point at index j of the points array belongs to the subset at index i of our cover. 
'''
class uniform_cover:
    def __init__(self, resolution, gain) -> None:
        self.resolution = resolution
        self.gain = gain

    def __call__(self, filtered_points) -> np.array:
        # determine the dimensionality of the filter output
        num_dimensions = filtered_points.shape[1] if len(filtered_points.shape) > 1 else 1

        # initialize min and max values for each dimension
        min_values = np.min(filtered_points, axis=0)
        max_values = np.max(filtered_points, axis=0)

        # initialize stride lengths for each dimension
        stride_lengths = np.array([self.resolution - self.resolution * self.gain])
        if len(stride_lengths.shape) > 1:
            stride_lengths = np.squeeze(stride_lengths)
        cover = []
    
        # initialize left and right endpoints for each dimension
        left_endpoints = min_values.T - np.finfo(float).eps
        right_endpoints = min_values.T + self.resolution
        if len(left_endpoints.shape) > 1:
            left_endpoints = np.squeeze(left_endpoints)
        if len(right_endpoints.shape) > 1:
            right_endpoints = np.squeeze(right_endpoints)

        # iterate over each dimension
        for dim in range(num_dimensions):
            while right_endpoints[dim] <= max_values[dim]:
                # boolean array representing whether a point belongs to this set in the cover
                cover_array = (filtered_points[:, dim] > left_endpoints[dim]) & (filtered_points[:, dim] < right_endpoints[dim]).astype(int)

                # add the set to our cover
                cover.append(cover_array)

                # shift window for this dimension
                left_endpoints[dim] += stride_lengths[dim]
                right_endpoints[dim] += stride_lengths[dim]

        # return cover
        return np.array(cover)

class threshold_cover:
    def __init__(self, thresholds) -> None:
        self.thresholds = thresholds
    def __call__(self, filtered_points) -> np.array:
        # determine the dimensionality of the filter output
        num_dimensions = filtered_points.shape[1] if len(filtered_points.shape) > 1 else 1

        # empty cover
        cover = []

        # for each dim get points within the thresholds and assign to the subset for that dim
        for dim in range(num_dimensions):
            subset = [((filtered_points[:, dim] >= low) & (filtered_points[:, dim] < high)).astype(int) for low, high in self.thresholds]
            cover.append(subset)
        
        return np.array(cover)

class dbscan_cover:
    def __init__(self, epsilon, neighbors, metric) -> None:
        self.epsilon = epsilon
        self.neighbors = neighbors
        self.metric = metric
    def __call__(self, filtered_points) -> np.array:
        clusters = DBSCAN(filtered_points, self.epsilon, self.neighbors, self.metric).clustered_indices
        cover = []
        for cluster in clusters:
            subset = [0 for i in range(len(filtered_points))]
            for point in cluster:
                subset[point] = 1
            cover.append(subset)

        return np.array(cover)