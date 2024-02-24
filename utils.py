import numpy as np 

def get_distances(points, point, metric):
    # gets the distances between a point and a set of points (i.e. a single row of the pairwise distance matrix)
    if metric == 'euclidean':
        distances = np.sqrt(np.sum(np.square(points - point), axis=-1))
    elif metric == 'cosine':
        # computing matrix product, transposing points for correct shapes
        dot_product = np.dot(point, points.T)
        # computing norms and product of norms
        norm_product = np.linalg.norm(point) * np.linalg.norm(points, axis=-1)
        # computing shifted cosine similarity then subtracting from 1 to convert to a distance
        distances = 1 - (((dot_product / norm_product) + 1) / 2)
    elif metric == 'geodesic':
        dot_products = np.dot(points, point)

        # compute the norm of each point and the reference_point
        norm_points = np.linalg.norm(points, axis=-1)
        norm_reference = np.linalg.norm(point)

        # compute the angular separation (arccos of the dot product divided by the product of the norms)
        angular_separation = np.arccos(np.clip(dot_products / (norm_points * norm_reference), -1, 1))

        # assuming that our points are on a hypersphere (i.e. radius = 1)
        radius = 1.0

        # compute geodesic distances (arc length on the hypersphere)
        distances = radius * angular_separation
    elif metric == 'manhattan':
        absolute_differences = np.abs(points - point)
        distances = np.sum(absolute_differences, axis=1)
    
    else:
        raise ValueError("metric must be one of the following: euclidean, cosine, geodesic, manhattan")
    
    
    return distances

def get_adjacency(points, metric, epsilon, neighbors=None, exclude_self=True):
    # this function iteratively constructs and returns an adjacency matrix using the chosen metric
    # zero matrix to start
    adjacency = np.zeros((len(points), len(points)))
    
    # for each point
    for i, point in enumerate(points):
        # compute the distance between point and all other points
        distances = get_distances(points, point, metric)
        # convert row vector into a boolean adjacency vector using our chosen epsilon
        if exclude_self:
            adjacency[i] = (distances > 0) & (distances < epsilon)
        else:
            adjacency[i] = (distances < epsilon)

        # remove points with degree less than the min neighbors threshold
        if neighbors:
            if np.sum(adjacency[i]) < neighbors:
                adjacency[i] = 0
    return adjacency