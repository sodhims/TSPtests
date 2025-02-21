import numpy as np
from scipy.stats import chi2
import json

class TSPPointGenerator:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        
    def uniform_random(self, n_points):
        """Generate uniform random points."""
        x = np.random.uniform(0, self.width, n_points)
        y = np.random.uniform(0, self.height, n_points)
        return np.column_stack((x, y))
    
    def clustered(self, n_points, n_clusters=3, cluster_std=10):
        """Generate clustered points using Gaussian distribution."""
        points = []
        points_per_cluster = n_points // n_clusters
        
        # Generate cluster centers
        centers = self.uniform_random(n_clusters)
        
        for center in centers:
            # Generate points around each center
            cluster_points_x = np.random.normal(center[0], cluster_std, points_per_cluster)
            cluster_points_y = np.random.normal(center[1], cluster_std, points_per_cluster)
            
            # Clip to bounds
            cluster_points_x = np.clip(cluster_points_x, 0, self.width)
            cluster_points_y = np.clip(cluster_points_y, 0, self.height)
            
            points.extend(zip(cluster_points_x, cluster_points_y))
        
        # Add remaining points if any
        remaining = n_points - len(points)
        if remaining > 0:
            extra_points = self.uniform_random(remaining)
            points.extend(zip(extra_points[:, 0], extra_points[:, 1]))
        
        return np.array(points)
    
    def perturbed_grid(self, n_points, noise_level=0.5):
        """Generate points on a grid with random perturbation."""
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_points)))
        x_step = self.width / grid_size
        y_step = self.height / grid_size
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_points:
                    # Add noise to grid positions
                    x = i * x_step + np.random.normal(0, noise_level * x_step)
                    y = j * y_step + np.random.normal(0, noise_level * y_step)
                    
                    # Clip to bounds
                    x = np.clip(x, 0, self.width)
                    y = np.clip(y, 0, self.height)
                    
                    points.append([x, y])
        
        return np.array(points)
    
    def radial(self, n_points, radial_std=0.3):
        """Generate points with radial distribution from center."""
        # Center point
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Generate radial coordinates
        radii = chi2.rvs(df=2, size=n_points) * radial_std * min(self.width, self.height) / 2
        angles = np.random.uniform(0, 2*np.pi, n_points)
        
        # Convert to Cartesian coordinates
        x = center_x + radii * np.cos(angles)
        y = center_y + radii * np.sin(angles)
        
        # Clip to bounds
        x = np.clip(x, 0, self.width)
        y = np.clip(y, 0, self.height)
        
        return np.column_stack((x, y))
    
    def mixed(self, n_points, cluster_ratio=0.6, radial_ratio=0.3):
        """Generate points using mixed distributions."""
        n_cluster = int(n_points * cluster_ratio)
        n_radial = int(n_points * radial_ratio)
        n_uniform = n_points - n_cluster - n_radial
        
        # Generate points using different methods
        cluster_points = self.clustered(n_cluster)
        radial_points = self.radial(n_radial)
        uniform_points = self.uniform_random(n_uniform)
        
        # Combine all points
        points = np.vstack((cluster_points, radial_points, uniform_points))
        
        # Shuffle the points
        np.random.shuffle(points)
        
        return points

    def save_points_to_json(self, points, distribution_type, instance_num):
        """Save points to JSON file with descriptive name."""
        # Create filename with pattern: tsp_distribution_npoints_instance.json
        n_points = len(points)
        filename = f"tsp_{distribution_type}_{n_points}_pts_{instance_num:03d}.json"
        
        # Format data according to the required structure
        data = {
            "cities": [{"x": float(x), "y": float(y)} for x, y in points]
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        return filename

# Example usage:
if __name__ == "__main__":
    # Create generator
    generator = TSPPointGenerator(width=100, height=100)
    
    # Set instance number
    instance = 42
    
    # Generate and save different point sets
    points_uniform = generator.uniform_random(20)
    generator.save_points_to_json(points_uniform, "uniform", instance)
    
    points_clustered = generator.clustered(20)
    generator.save_points_to_json(points_clustered, "clustered", instance)
    
    points_grid = generator.perturbed_grid(20)
    generator.save_points_to_json(points_grid, "grid", instance)
    
    points_radial = generator.radial(20)
    generator.save_points_to_json(points_radial, "radial", instance)
    
    points_mixed = generator.mixed(20)
    generator.save_points_to_json(points_mixed, "mixed", instance)