Mapper is an algorithm for performing topological data analysis of point clouds which takes the following parameters:

let X represent the input space

a filter function f: X -> F where F represents sum subspace of R^N of lower dimension than X

a cover function c: F -> C where C is a cover of F

epsilon, min_neighbors, and a metric for DBSCAN clustering of the preimages of the cover function

the output is called the nerve, which is a graph representing intersections between the covering sets

the algorithm takes the following steps:
1. map points from X to F with the filtering function
2. generate a covering of the outputs of the filter function
3. cluster the preimage of each set in the cover using dbscan. each of these clusters will be a vertex in the nerve. these vertices are also color coded by which set in the cover they belong to
4. for each cluster c1 in the nerve if another cluster c2 has a non empty intersection then we draw an edge between vertices c1 and c2

This implementation can be expanded in a variety of ways.

1. custom filters can be implemented by simply making a callable object that takes in an N x M matrix of points and returns an N x K matrix of filter values.
2. custom covers are also implemented as callable objects. a cover map must take in an N x K matrix of filter values and output a C x N matrix where each row represents a set in the cover and each column represents the index of a particular point. 
Each row is a boolean mask that represents whether a point belongs to a set in the cover or not. 
A value of 1 in position i, j means that the point at row j in the input points array belongs to set i of the cover
3. metrics can be added to utils.py in the get_distances function by simply adding another condition to the elif block. the metric should compute distances between a single point and a list of points.

Requirements Notes:
matplotlib is only required in the examples.ipynb but is not required to use mapper
