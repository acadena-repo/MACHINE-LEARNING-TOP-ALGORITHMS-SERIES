# PROGRAMMER: Alejandro Cadena
# DATE CREATED: 10.03.2020                   
# REVISED DATE: 
# K-MEANS++ ALGORITHM
# PURPOSE: The K-means++ algorithm chooses one point from the data as the first cluster center.
# The next cluster centers are chosen from the remaining points. The probability of picking any other point 
# is proportional to the square of the euclidean distance of that point to the nearest cluster center.
#
import numpy as np

def init_clusters(x_points, no_clusters = 3):
    # Create List to store clusters
    clusters = []
    
    # Save list of cluster indicies
    arr_idx = np.arange(len(x_points))
    
    # Choose first cluster; append to list
    clusters.append( x_points[np.random.choice(arr_idx)])
    
    # Define function to calculate squared distance
    def dist_sq(x): return np.linalg.norm(x)**2
    
    c_dist = None

    # Add Clusters until reaching "num_clusters"
    while len(clusters) < no_clusters:
        
        # Calculate distance between latest cluster and rest of points
        new_dist = np.apply_along_axis(np.linalg.norm, 1, x_points - clusters[-1]).reshape(-1,1)
        
        # Add to distance array - First check to see if distance matrix exists
        if type(c_dist) == type(None):
            c_dist = new_dist
            
        else:
            c_dist = np.concatenate([c_dist, new_dist], axis = 1)
        
        # Calculate probability by finding shortest distance, then normalizing
        c_prob = np.apply_along_axis(np.min, 1, c_dist)
        c_prob = c_prob / c_prob.sum()

        # Draw new cluster according to probability
        clusters.append(x_points[np.random.choice(arr_idx, p = c_prob)])
            
    return clusters

def threshold_monitor(n_centroids, o_centroids, threshold):
    n_centroids = np.array(n_centroids)
    o_centroids = np.array(o_centroids)
    
    for n, o in zip(n_centroids, o_centroids):
        if np.linalg.norm(n-o) > threshold:
            return False
    return True