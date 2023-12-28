import math
from shapely.geometry.polygon import Polygon

class ClusterHelper:
    def __init__(self):
        self.capacity_truck = 20  # Capacity from your thesis
    
    def evaluate_cluster(self, cluster_list):
        most_left = 999
        most_right = -1
        most_top = -1
        most_bottom = 999
        sum_lat = 0
        sum_lon = 0
    
        for location in cluster_list:
            lat, lon = location
            most_left = min(most_left, lon)
            most_right = max(most_right, lon)
            most_top = max(most_top, lat)
            most_bottom = min(most_bottom, lat)
            sum_lat += lat
            sum_lon += lon
    
        cluster_centroid = (sum_lat / len(cluster_list), sum_lon / len(cluster_list))
        
        # Calculate cluster_midpoint
        cluster_midpoint = ((most_top + most_bottom) / 2, (most_left + most_right) / 2)
        
        cluster_length = most_top - most_bottom
        cluster_width = most_right - most_left
        area = cluster_length * cluster_width
        shape = min(cluster_length, cluster_width) / max(cluster_length, cluster_width)
        
        if shape <= 0.6:
            shape_constant = 0.55
        else:
            shape_constant = 0.45
        
        daganzo = 0.9 * shape_constant * (len(cluster_list) / (self.capacity_truck ** 2)) * (math.sqrt(area * len(cluster_list)))
        
        return most_left, most_right, most_top, most_bottom, cluster_centroid, cluster_midpoint, cluster_length, cluster_width, area, shape, daganzo


    def create_polygons(self, cluster_width, cluster_length, most_left, most_bottom, polygon_nr):
        steps_hor = (cluster_width / polygon_nr) * 1.001
        steps_vert = (cluster_length / polygon_nr) * 1.001
        start_left = most_left - 0.00001
        start_bottom = most_bottom - 0.00001
        polygons = []

        for vert in range(polygon_nr):
            for hor in range(polygon_nr):
                polygon = Polygon([
                    ((start_left + (hor * steps_hor)), (start_bottom + (vert * steps_vert))),
                    ((start_left + ((hor + 1) * steps_hor)), (start_bottom + (vert * steps_vert))),
                    ((start_left + ((hor + 1) * steps_hor)), (start_bottom + ((vert + 1) * steps_vert))),
                    ((start_left + (hor * steps_hor)), (start_bottom + ((vert + 1) * steps_vert)))
                ])
                polygons.append(polygon)

        return polygons

    def bearing_between_points_radian(self, point_a, point_b):
        lat1, lon1 = point_a
        lat2, lon2 = point_b
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        diff_lon = math.radians(lon2 - lon1)
        x = math.sin(diff_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        radian_bearing = compass_bearing * (math.pi / 180)
        return radian_bearing