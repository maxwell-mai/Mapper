import numpy as np
from itertools import combinations
from utils import get_distances, get_adjacency
from math import ceil

class coordinate_filter:
    def __init__(self, dims) -> None:
        self.dims = dims
    def __call__(self, points) -> np.array:
        # this function filters by pulling values from only a single dimension or set of dimensions
        # getting values along the desired axes for each point
        coord_projection = points[:, self.dims]
        return coord_projection

class distance_filter:
    def __init__(self, reference_point, metric) -> None:
        self.reference_point = reference_point
        self.metric = metric
    def __call__(self, points) -> np.array:
        distances = get_distances(points, self.reference_point, self.metric)
        return distances

class local_stat_filters:
    def __init__(self, measure, window_size, overlap) -> None:
        self.measure = measure
        self.window_size = window_size
        self.overlap = overlap
    def __call__(self, points) -> np.array:
        # computes mean of points in overlapping windows of the points array. There will always be n means for n points
        n = len(points)
        step_size = int(self.window_size * (1-self.overlap))

        if self.measure == 'mean':
            measure_func = np.mean
        elif self.measure == 'median':
            measure_func = np.median
        elif self.measure == 'variance':
            measure_func = np.var
        elif self.measure == 'std':
            measure_func = np.std
        else:
            raise ValueError("measure must be one of the following: mean, median, variance, std")

        local_stats = np.zeros(n)

        for i in range(0, n - self.window_size + 1, step_size):
            local_stats[i:i+self.window_size] = measure_func(points[i:i+self.window_size])    

        return local_stats.reshape((n, 1))

class density_filter:
    def __init__(self, metric, neighbors, normalize=False) -> None:
        self.metric = metric
        self.neighbors = neighbors
        self.normalize = normalize
    def __call__(self, points) -> np.array:
        local_densities = np.zeros(len(points))
        for index, point in enumerate(points):
            distances = get_distances(points, point, self.metric)
            local_neighbors = np.argsort(distances)[-self.neighbors:]
            local_density = 1 / np.mean(np.array([distances[i] for i in local_neighbors]))
            local_densities[index] = local_density

        if self.normalize:
            local_densities = (local_densities - np.min(local_densities)) / (np.max(local_densities) - np.min(local_densities))

        return local_densities.reshape((len(points), 1))


class homology_filter:
    def __init__(self, measure, metric, epsilon, max_dims, step_size=None, max_steps=None) -> None:
        # this class is just to group up the methods required for homology filters based on the vietoris rips complex
        self.measure = measure
        self.metric = metric
        self.epsilon = epsilon
        self.max_dims = max_dims
        # step_size is only necessary for the persistence interval filter
        self.step_size = step_size
        self.max_steps = max_steps
    
    def __call__(self, points) -> np.array:
        if self.measure == 'persistence':
            if not self.step_size:
                raise(ValueError('You must include a step_size in order to use the persistence filter'))
            return self.compute_persistence_intervals(points)
        elif self.measure == 'distance':
            return self.compute_distance_to_nearest_simplex(points)
    def get_vietoris_rips_complex(self, points, metric, filter_val, max_dims) -> dict:
        n = len(points)

        # dict with simplex dimension as keys and list of simplices as values. each simplex in the list is an n tuple of integers where n is the dimension + 1 and the integers represent row indeces of the points array
        vietoris_rips_complex = {}

        # for each of the desired dimensions
        for dim in range(1, max_dims + 1):
            simplices = set()        
            # Iterate over all combinations of simplex vertices to find simplices of the desired dimension
            for simplex_vertices in combinations(range(n), dim + 1):
                simplex_points = points[simplex_vertices, :]
                # getting a local adjacency matrix by extracting only the rows and columns associated with the simplex vertices
                local_adjacency = get_adjacency(simplex_points, metric, filter_val, exclude_self=False)
                # Check if the simplex is present in the adjacency matrix (all edges exist)
                if np.all(local_adjacency == 1):
                    simplices.add(simplex_vertices)

            points_index = [0 for i in range(len(points))]
            for i, simplex in enumerate(simplices):
                for vertex in simplex:
                    points_index[vertex] = i

            vietoris_rips_complex[dim] = {'simplices': simplices, 'points_index' : points_index}
        return vietoris_rips_complex
    
    def compute_persistence_intervals(self, points) -> np.array:
        # getting initial complex using provided epsilon
        base_complex = self.get_vietoris_rips_complex(points, self.metric, self.epsilon, self.max_dims)

        # multi level dictionary with first level keys equal to simplex dimension, second level keys equal to simplex vertex tuples, and values equal to 2 tuples of form (birth, death)
        persistence_intervals = {i : {} for i in range(1, self.max_dims+1)}
        # setting birth values for each initial n-simplex 
        for dim in base_complex:
            for simplex in base_complex[dim]['simplices']:
                persistence_intervals[dim][simplex] = (self.epsilon, 0)
        
        filter_val = self.epsilon + self.step_size
        steps = 0
        while (not self.all_simplices_dead(persistence_intervals)) and (steps <= self.max_steps):
            # recomputing vietoris rips complex with new filter value
            vietoris_rips_complex = self.get_vietoris_rips_complex(points, self.metric, filter_val, self.max_dims)

            for dim in vietoris_rips_complex:
                # checking for newly born simpleces in this dim
                for simplex in vietoris_rips_complex[dim]['simplices']:
                    if simplex not in persistence_intervals[dim]:
                        persistence_intervals[dim][simplex] = (filter_val, 0)
                # checking for newly dead simpleces in this dim
                for simplex in persistence_intervals[dim]:
                    if simplex not in vietoris_rips_complex[dim]['simplices']:
                        persistence_intervals[dim][simplex][1] = filter_val
            
            filter_val += self.step_size
            steps += 1

        print(persistence_intervals)
        
        persistence_interval_filter = []
        # for each point
        for index, point in enumerate(points):
            average_ages = []
            # for each dimension
            for dim in persistence_intervals:
                min_birth = 10**6
                death = None
                # for each simplex in that dimension
                for simplex in persistence_intervals[dim]:
                    # if the point is one of the vertices of the simplex
                    if index in simplex:
                        # we are associating a point to the youngest simplex which contains it in each dimension
                        if persistence_intervals[dim][simplex][0] < min_birth:
                            min_birth = persistence_intervals[dim][simplex][0]
                            if persistence_intervals[dim][simplex][1] == 0:
                                death = filter_val
                            else:
                                death = persistence_intervals[dim][simplex][1]

                # computing our actual filtered value for the point
                if death:
                    average_ages.append((death - min_birth) / 2)
                else:
                    average_ages.append(0)
            persistence_interval_filter.append(average_ages)
        
        # persistence interval filter is of shape n, m where n is the number of points and m is the number of dimensions
        return np.array(persistence_interval_filter)
                        

    def all_simplices_dead(self, persistence_intervals) -> bool:
        for dim in persistence_intervals:
            for simplex in persistence_intervals[dim]:
                if persistence_intervals[dim][simplex][1] == 'inf':
                    return False
        
        return True
    
    def compute_distance_to_nearest_simplex(self, points) -> np.array:
        vietoris_rips_complex = self.get_vietoris_rips_complex(points, self.metric, self.epsilon, self.max_dims)
        dist_to_simplex_filter = []    
        
        for i, point in enumerate(points):
            distances = []
            for dim in vietoris_rips_complex:
                simplices = vietoris_rips_complex[dim]['simplices']
                min_dist = 10**6
                for simplex in simplices:
                    if i not in simplex:
                        ref_point = points[simplex[0]]
                        dist = get_distances([ref_point], point, self.metric)
                        if dist < min_dist:
                            min_dist = dist
                distances.append(min_dist)
            dist_to_simplex_filter.append(distances)

        return np.squeeze(np.array(dist_to_simplex_filter))




    

    
    
