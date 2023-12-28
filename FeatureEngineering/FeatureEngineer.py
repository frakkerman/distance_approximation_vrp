import pandas as pd
import numpy as np
import statistics
from scipy.spatial import ConvexHull, distance
from shapely.geometry import Point
from ClusterHelper import ClusterHelper
import math
import concurrent.futures
# from haversine import haversine (can be used as distance function instead of distance.euclidean)

class FeatureEngineer:
    def __init__(self, raw_data, start_location=None, polygon_nr1=10, polygon_nr2=15, circle_size_1=0.5, circle_size_2=0.75, vehicle_capacity=8000):
        self.raw_data = raw_data
        self.start_location = start_location
        self.polygon_nr1 = polygon_nr1
        self.polygon_nr2 = polygon_nr2
        self.polygon_1 = None
        self.polygon_2 = None
        self.circle_size_1 = circle_size_1
        self.circle_size_2 = circle_size_2
        self.polygon_counter_1 = []
        self.polygon_counter_2 = []
        self.dist_from_start_list = []
        self.dist_from_centroid_list = []
        self.dist_from_midpoint_list = []
        self.rad_from_start = []
        self.rad_from_centroid = []
        self.rad_from_midpoint = []
        self.vehicle_capacity = vehicle_capacity
        self.new_data = pd.DataFrame()  # Initialize an empty DataFrame for new data
        self.distance_function = distance.euclidean
        self.func = ClusterHelper()
        self.bearing_function = self.func.bearing_between_points_radian
        


    def _process_group(self, cluster_id, group):
        # Skip the placeholder cluster ID
        if cluster_id == 10000000000:
            return None

        # Rest of processing logic
        ClusterList = group[["Lat", "Lon"]].values.tolist()
        DemandList = group["ExpFillLevel"].tolist()
        Distance = group["Distance"].iloc[0]
        ServiceLevel = group["ServiceLevel"].iloc[0]

        features = self._calculate_features(ClusterList, DemandList, Distance, ServiceLevel)
        features['Target'] = self.calculate_target_feature(group)

        return features

    def process_data(self):
        grouped_data = self.raw_data.groupby('ClusterID')
        futures = []
        results = []

        # Use ProcessPoolExecutor for CPU-bound tasks or ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for cluster_id, group in grouped_data:
                # Submit the processing function to the executor
                future = executor.submit(self._process_group, cluster_id, group)
                futures.append(future)

            # Collect results as they are completed
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Concatenate all results into the new_data DataFrame
        self.new_data = pd.concat([self.new_data, pd.DataFrame(results)], ignore_index=True)

    def get_new_data(self):
        return self.new_data
   

    def _calculate_features(self, ClusterList, DemandList, Distance, ServiceLevel):
        # Additional required data to be calculated or retrieved
        MostLeft,MostRight,MostTop,MostBottom,cluster_centroid,cluster_midpoint, cluster_length, \
        cluster_width,area,shape,daganzo = self.func.evaluate_cluster(ClusterList)#Function with all kind of metrics
        self.polygon_1 = self.func.create_polygons(cluster_width,cluster_length,MostLeft,MostBottom,self.polygon_nr1)#polygon creation
        self.polygon_2 = self.func.create_polygons(cluster_width,cluster_length,MostLeft,MostBottom,self.polygon_nr2)#polygon creation
        self.polygon_counter_1 = [0] * len(self.polygon_1)
        self.polygon_counter_2 = [0] * len(self.polygon_2)
        
        # Store cluster metrics as instance variables for later use
        self.cluster_metrics = {
           'MostLeft': MostLeft,
           'MostRight': MostRight,
           'MostTop': MostTop,
           'MostBottom': MostBottom,
           'ClusterCentroid': cluster_centroid,
           'ClusterMidpoint': cluster_midpoint,
           'ClusterLength': cluster_length,
           'ClusterWidth': cluster_width,
           'Area': area,
           'Shape': shape,
           'Daganzo': daganzo
           }
    
        # Calculating features
        demand_capacity_features = self.calculate_demand_capacity_features(DemandList, self.vehicle_capacity)
        cluster_features = self.calculate_cluster_features(ClusterList, self.polygon_nr1, self.polygon_nr2)
        distance_polygon_counts = self.calculate_distance_and_polygon_counts(ClusterList, self.start_location, cluster_centroid, cluster_midpoint, self.polygon_1, self.polygon_2, self.distance_function, self.bearing_function)
        polygon_features = self.calculate_polygon_features([self.polygon_counter_1, self.polygon_counter_2], [self.polygon_1, self.polygon_2], self.start_location, cluster_centroid, cluster_midpoint, self.distance_function)
        circle_counters = self.calculate_circle_counters(self.dist_from_centroid_list, self.dist_from_midpoint_list, self.dist_from_start_list, self.circle_size_1, self.circle_size_2)
        variances = self.calculate_variances(self.dist_from_start_list, self.dist_from_centroid_list, self.dist_from_midpoint_list, self.rad_from_start, self.rad_from_midpoint,self.rad_from_centroid)
        upper_triangle_metrics = self.calculate_upper_triangle_metrics(ClusterList, self.distance_function)
    
        calculated_features = {
            **demand_capacity_features,
            **cluster_features,
            **distance_polygon_counts,
            **polygon_features,
            **circle_counters,
            **variances,
            **upper_triangle_metrics,
            'Daganzo': daganzo  # Add the 'daganzo' feature
            # Add other feature calculations as needed
        }
    
        return calculated_features
    
    def calculate_demand_capacity_features(self,demand_list, vehicle_capacity):
     total_demand = np.sum(demand_list, dtype=float)
     avg_demand = total_demand / len(demand_list)
     var_demand = np.var(demand_list, dtype=float)
     cap_div_avg_dem = avg_demand / vehicle_capacity  # Nicola
     tot_dem_div_cap = total_demand / vehicle_capacity  # Nicola
     max_dem_div_cap = total_demand / vehicle_capacity  # Rasku
     min_trucks = math.ceil(tot_dem_div_cap)  # Rasku
     
     return {
         'TotalDemand': total_demand,
         'AvgDemand': avg_demand,
         'VarDemand': var_demand,
         'CapDivAvgDem': cap_div_avg_dem,
         'TotDemDivCap': tot_dem_div_cap,
         'MaxDemDivCap': max_dem_div_cap,
         'MinTrucks': min_trucks
     }


    def calculate_cluster_features(self, cluster_list, cluster_width, cluster_length):
        # Extract latitude and longitude lists
        lat_list = [location[0] for location in cluster_list]
        lon_list = [location[1] for location in cluster_list]
    
        # Calculate variances
        var_lat = statistics.variance(lat_list) if len(lat_list) > 1 else 0
        var_lon = statistics.variance(lon_list) if len(lon_list) > 1 else 0
        var_lat_lon = var_lat * var_lon
    
        # Compute convex hull properties
        if len(cluster_list) > 2:
            hull = ConvexHull(cluster_list)
            hull_perimeter = hull.area
            hull_area = hull.volume
        else:
            hull_perimeter = 0
            hull_area = 0
    
        # Access cluster metrics from instance variable
        cluster_metrics = self.cluster_metrics
    
        # Calculate Cluster Perimeter
        cluster_perimeter = 2 * cluster_metrics['ClusterWidth'] + 2 * cluster_metrics['ClusterLength']
    
        return {
            'VarLat': var_lat,
            'VarLon': var_lon,
            'VarLatLon': var_lat_lon,
            'HullPerimeter': hull_perimeter,
            'HullArea': hull_area,
            'ClusterLength': cluster_metrics['ClusterLength'],
            'ClusterWidth': cluster_metrics['ClusterWidth'],
            'ClusterArea': cluster_metrics['Area'],
            'ClusterPerimeter': cluster_perimeter,
            # ... any other metrics ...
        }

    def calculate_distance_and_polygon_counts(self, cluster_list, start_location, cluster_centroid, cluster_midpoint, polygon_1, polygon_2, distance_function, bearing_function):
        total_dist_from_start = 0
        total_dist_from_centroid = 0
        total_dist_from_midpoint = 0
    
        for location in cluster_list:
            curr_location = (location[0], location[1])  # Lat, Lon
    
            # Calculate distances
            total_dist_from_start += distance_function(start_location, curr_location)
            total_dist_from_centroid += distance_function(cluster_centroid, curr_location)
            total_dist_from_midpoint += distance_function(cluster_midpoint, curr_location)
    
            # Append to lists
            self.dist_from_start_list.append(distance_function(start_location, curr_location))
            self.dist_from_centroid_list.append(distance_function(cluster_centroid, curr_location))
            self.dist_from_midpoint_list.append(distance_function(cluster_midpoint, curr_location))
    
            # Calculate bearings
            self.rad_from_start.append(bearing_function(start_location, curr_location))
            self.rad_from_centroid.append(bearing_function(cluster_centroid, curr_location))
            self.rad_from_midpoint.append(bearing_function(cluster_midpoint, curr_location))
    
            # Polygon counts
            point = Point(location[1], location[0])  # x, y = lon, lat
            for i in range(len(polygon_1)):
                if polygon_1[i].contains(point):
                    self.polygon_counter_1[i] += 1
            for i in range(len(polygon_2)):
                if polygon_2[i].contains(point):
                    self.polygon_counter_2[i] += 1
    
        return {
            'TotalDistFromStart': total_dist_from_start,
            'TotalDistFromCentroid': total_dist_from_centroid,
            'TotalDistFromMidpoint': total_dist_from_midpoint,
            'avgRadFromStart': np.mean(self.rad_from_start),
            'avgRadFromCentroid': np.mean(self.rad_from_centroid),
            'avgRadFromMidpoint': np.mean(self.rad_from_midpoint),
            'avgPositivePolygonCounter1': sum(num for num in self.polygon_counter_1 if num > 0) / len([num for num in self.polygon_counter_1 if num > 0]) if any(num > 0 for num in self.polygon_counter_1) else 0,
            'avgPositivePolygonCounter2': sum(num for num in self.polygon_counter_2 if num > 0) / len([num for num in self.polygon_counter_2 if num > 0]) if any(num > 0 for num in self.polygon_counter_2) else 0
            # ... other metrics if needed ...
        }


    def calculate_polygon_features(self, polygon_counters, polygons, start_location, cluster_centroid, cluster_midpoint, distance_function):
        features = {}
    
        for idx, polygon_counter in enumerate(polygon_counters):
            nr_of_polygons_positive = sum(count > 0 for count in polygon_counter)
            active_polygon_list = [polygons[idx][i] for i, count in enumerate(polygon_counter) if count > 0]
    
            centroid_lons = [polygon.centroid.x for polygon in active_polygon_list]
            centroid_lats = [polygon.centroid.y for polygon in active_polygon_list]
            
            polygon_centroid_lon = sum(centroid_lons) / len(centroid_lons) if centroid_lons else 0
            polygon_centroid_lat = sum(centroid_lats) / len(centroid_lats) if centroid_lats else 0
    
            poly_centroid_all_active = (polygon_centroid_lat, polygon_centroid_lon)
            distance_start_all_polygons = distance_function(start_location, poly_centroid_all_active)
            distance_centroid_all_polygon = distance_function(cluster_centroid, poly_centroid_all_active)
            distance_midpoint_all_polygon = distance_function(cluster_midpoint, poly_centroid_all_active)
    
            avg_polygon_content = sum(polygon_counter) / len(polygon_counter)
            avg_positive_polygon_content = sum(polygon_counter) / nr_of_polygons_positive if nr_of_polygons_positive else 0
    
            # Distance from largest polygon to start location and others
            if polygon_counter:
                largest_polygon_index = polygon_counter.index(max(polygon_counter))
                largest_polygon_centroid = polygons[idx][largest_polygon_index].centroid.coords[0]
                poly_centroid_coord = (largest_polygon_centroid[1], largest_polygon_centroid[0])
    
                distance_start_largest_polygon = distance_function(start_location, poly_centroid_coord)
                distance_centroid_largest_polygon = distance_function(cluster_centroid, poly_centroid_coord)
                distance_midpoint_largest_polygon = distance_function(cluster_midpoint, poly_centroid_coord)
            else:
                distance_start_largest_polygon = distance_centroid_largest_polygon = distance_midpoint_largest_polygon = 0
    
            # Distance from largest polygon to all active polygon centroid
            distance_largest_all = distance_function(poly_centroid_coord, poly_centroid_all_active) if polygon_counter else 0
    
            # Storing the results
            prefix = f'Polygon{idx + 1}_'
            features[prefix + 'NrOfPolygonsPositive'] = nr_of_polygons_positive
            features[prefix + 'AvgContent'] = avg_polygon_content
            features[prefix + 'AvgPositiveContent'] = avg_positive_polygon_content
            features[prefix + 'DistanceStartAll'] = distance_start_all_polygons
            features[prefix + 'DistanceCentroidAll'] = distance_centroid_all_polygon
            features[prefix + 'DistanceMidpointAll'] = distance_midpoint_all_polygon
            features[prefix + 'DistanceStartLargest'] = distance_start_largest_polygon
            features[prefix + 'DistanceCentroidLargest'] = distance_centroid_largest_polygon
            features[prefix + 'DistanceMidpointLargest'] = distance_midpoint_largest_polygon
            features[prefix + 'DistanceLargestAll'] = distance_largest_all
    
        return features

   
    def calculate_circle_counters(self,dist_from_centroid_list, dist_from_midpoint_list, dist_from_start_list,circle_size_1,circle_size_2):
        circle_ray_centroid = max(dist_from_centroid_list)
        counter_in_circle_centroid_1 = sum(dist <= circle_ray_centroid * circle_size_1 for dist in dist_from_centroid_list)
        counter_in_circle_centroid_2 = sum(dist <= circle_ray_centroid * circle_size_2 for dist in dist_from_centroid_list)
    
        circle_ray_midpoint = max(dist_from_midpoint_list)
        counter_in_circle_midpoint_1 = sum(dist <= circle_ray_midpoint * circle_size_1 for dist in dist_from_midpoint_list)
        counter_in_circle_midpoint_2 = sum(dist <= circle_ray_midpoint * circle_size_2 for dist in dist_from_midpoint_list)
    
        circle_ray_startpoint = max(dist_from_start_list)
        counter_in_circle_startpoint_1 = sum(dist <= circle_ray_startpoint * circle_size_1 for dist in dist_from_start_list)
        counter_in_circle_startpoint_2 = sum(dist <= circle_ray_startpoint * circle_size_2 for dist in dist_from_start_list)
    
        return {
            'CounterInCircleCentroid1': counter_in_circle_centroid_1,
            'CounterInCircleCentroid2': counter_in_circle_centroid_2,
            'CounterInCircleMidpoint1': counter_in_circle_midpoint_1,
            'CounterInCircleMidpoint2': counter_in_circle_midpoint_2,
            'CounterInCircleStartpoint1': counter_in_circle_startpoint_1,
            'CounterInCircleStartpoint2': counter_in_circle_startpoint_2
        }

    def calculate_variances(self, *lists):
        variances = {f'Variance_{i}': statistics.variance(lst) for i, lst in enumerate(lists) if lst}
        return variances

    def calculate_upper_triangle_metrics(self, cluster_list, distance_function):
        # Create a distance matrix
        num_locations = len(cluster_list)
        distance_matrix = np.zeros((num_locations, num_locations))
    
        for i in range(num_locations):
            for j in range(i + 1, num_locations):
                distance_matrix[i, j] = distance_function(cluster_list[i], cluster_list[j])
                distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix
    
        # Extract the upper triangle values excluding the diagonal
        upper_triangle = distance_matrix[np.triu_indices(num_locations, k=1)]
    
        # Calculate metrics
        avg_dist_location_pairs = np.mean(upper_triangle)
        var_dist_location_pairs = np.var(upper_triangle)
    
        return {'AvgDistLocationPairs': avg_dist_location_pairs, 'VarDistLocationPairs': var_dist_location_pairs, 'num_locations':num_locations}


    def calculate_distance_matrix(self,cluster_list, distance_function):
        matrix_size = len(cluster_list)
        distance_matrix = np.zeros((matrix_size, matrix_size))
    
        for i in range(matrix_size):
            for j in range(matrix_size):
                point_i = (cluster_list[i][0], cluster_list[i][1])
                point_j = (cluster_list[j][0], cluster_list[j][1])
                distance_matrix[i, j] = distance_function(point_i, point_j)
    
        return distance_matrix
    
    def calculate_target_feature(self, group):
        """
        Get the target feature for a cluster based on the 'Distance' value.
        
        :param group: DataFrame group representing a cluster
        :return: The 'Distance' value from any entry in the cluster group
        """
        return group['Distance'].iloc[0]