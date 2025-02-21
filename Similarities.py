import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
import tkinter as tk
from tkinter import filedialog
import json

def select_files():
    """Open file dialog to select two JSON files."""
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the first JSON file...")
    file1 = filedialog.askopenfilename(
        title="Select first JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if not file1:
        print("No file selected. Exiting...")
        return None, None
    
    print("Please select the second JSON file...")
    file2 = filedialog.askopenfilename(
        title="Select second JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if not file2:
        print("No file selected. Exiting...")
        return None, None
    
    return file1, file2

def load_points_from_json(filename):
    """Load points from JSON file and convert to numpy array."""
    with open(filename, 'r') as f:
        data = json.load(f)
    points = np.array([(city['x'], city['y']) for city in data['cities']])
    return points

# Traditional Similarity Measures

def hausdorff_distance(points1, points2):
    """Compute the Hausdorff distance between two point sets."""
    distances = cdist(points1, points2)
    h1 = np.max(np.min(distances, axis=1))
    h2 = np.max(np.min(distances, axis=0))
    return max(h1, h2)

def earth_movers_distance(points1, points2):
    """Compute the Earth Mover's Distance (1D approximation for each dimension)."""
    x1, y1 = np.sort(points1[:, 0]), np.sort(points1[:, 1])
    x2, y2 = np.sort(points2[:, 0]), np.sort(points2[:, 1])
    
    if len(x1) != len(x2):
        size = max(len(x1), len(x2))
        x1_interp = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(x1)), x1)
        x2_interp = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(x2)), x2)
        y1_interp = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(y1)), y1)
        y2_interp = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(y2)), y2)
    else:
        x1_interp, x2_interp, y1_interp, y2_interp = x1, x2, y1, y2
    
    emd_x = wasserstein_distance(x1_interp, x2_interp)
    emd_y = wasserstein_distance(y1_interp, y2_interp)
    return np.sqrt(emd_x**2 + emd_y**2)

def procrustes_distance(points1, points2):
    """Compute the Procrustes distance between two point sets."""
    if len(points1) != len(points2):
        size = max(len(points1), len(points2))
        x1 = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(points1)), points1[:, 0])
        y1 = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(points1)), points1[:, 1])
        x2 = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(points2)), points2[:, 0])
        y2 = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(points2)), points2[:, 1])
        points1_interp = np.column_stack((x1, y1))
        points2_interp = np.column_stack((x2, y2))
    else:
        points1_interp, points2_interp = points1, points2
    
    _, _, disparity = procrustes(points1_interp, points2_interp)
    return disparity

def mutual_information_approximation(points1, points2, bins=20):
    """Compute an approximation of mutual information using 2D histograms."""
    hist1, _, _ = np.histogram2d(points1[:, 0], points1[:, 1], bins=bins)
    hist2, _, _ = np.histogram2d(points2[:, 0], points2[:, 1], bins=bins)
    
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    eps = 1e-10
    mi = np.sum(hist1 * np.log((hist1 + eps) / (hist2 + eps)))
    return mi

def frechet_distance_approximation(points1, points2):
    """Compute an approximation of the Frechet distance."""
    distances = cdist(points1, points2)
    row_ind, col_ind = linear_sum_assignment(distances)
    return np.max(distances[row_ind, col_ind])

# Spectral Similarity Measures

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
import tkinter as tk
from tkinter import filedialog
import json


# Spectral Similarity Measures

def compute_adjacency_matrix(points, sigma=None):
    """Compute Gaussian adjacency matrix for points with adaptive sigma."""
    distances = cdist(points, points)
    
    # Adaptive sigma based on mean distance if not provided
    if sigma is None:
        sigma = np.mean(distances) / 2
    
    # Avoid numerical instability with very small distances
    A = np.exp(-distances**2 / (2 * sigma**2))
    
    # Ensure matrix is not singular
    A += 1e-8 * np.eye(len(points))
    return A

def compute_laplacian_spectrum(points, k=10):
    """Compute the first k eigenvalues of the normalized Laplacian."""
    # Compute adjacency matrix with adaptive sigma
    A = compute_adjacency_matrix(points)
    
    # Compute degree matrix
    d = np.sum(A, axis=1)
    D = np.diag(d)
    
    # Avoid division by zero in normalization
    d_inv_sqrt = 1 / np.sqrt(d + 1e-8)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Compute normalized Laplacian
    L = np.eye(len(points)) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Ensure symmetry (might be broken by numerical errors)
    L = (L + L.T) / 2
    
    # Compute eigenvalues and sort them
    eigenvals = linalg.eigvalsh(L)
    eigenvals = np.sort(np.abs(eigenvals))[:k]  # Take absolute values
    
    return eigenvals

def spectral_distance(points1, points2, k=10):
    """Compute distance between Laplacian spectra."""
    spec1 = compute_laplacian_spectrum(points1, k)
    spec2 = compute_laplacian_spectrum(points2, k)
    
    # Normalize spectra before comparison
    spec1_norm = spec1 / (np.linalg.norm(spec1) + 1e-8)
    spec2_norm = spec2 / (np.linalg.norm(spec2) + 1e-8)
    
    return np.linalg.norm(spec1_norm - spec2_norm)

def heat_trace_signature(points, t=None):
    """Compute heat trace signature for different time values."""
    if t is None:
        # Adaptive time scales based on spectrum
        spectrum = compute_laplacian_spectrum(points, k=len(points))
        t_min = 1.0 / (np.max(spectrum) + 1e-8)
        t_max = 1.0 / (np.min(spectrum[1:] + 1e-8))  # Skip first eigenvalue
        t = np.logspace(np.log10(t_min), np.log10(t_max), 10)
    
    spectrum = compute_laplacian_spectrum(points, k=min(len(points), 20))
    heat_trace = np.array([np.sum(np.exp(-t_i * spectrum)) for t_i in t])
    
    # Normalize the heat trace
    heat_trace = heat_trace / (np.linalg.norm(heat_trace) + 1e-8)
    return heat_trace

def heat_trace_distance(points1, points2):
    """Compute distance between heat trace signatures."""
    # Use same time points for both signatures
    spectrum1 = compute_laplacian_spectrum(points1, k=len(points1))
    spectrum2 = compute_laplacian_spectrum(points2, k=len(points2))
    
    t_min = max(1.0 / (np.max(spectrum1) + 1e-8), 1.0 / (np.max(spectrum2) + 1e-8))
    t_max = min(1.0 / (np.min(spectrum1[1:] + 1e-8)), 1.0 / (np.min(spectrum2[1:] + 1e-8)))
    t = np.logspace(np.log10(t_min), np.log10(t_max), 10)
    
    ht1 = heat_trace_signature(points1, t)
    ht2 = heat_trace_signature(points2, t)
    return np.linalg.norm(ht1 - ht2)

def wave_trace_signature(points, t=None):
    """Compute wave trace signature for different time values."""
    if t is None:
        # Adaptive time scales
        spectrum = compute_laplacian_spectrum(points, k=min(len(points), 20))
        t_max = 2 * np.pi / (np.min(np.sqrt(spectrum[1:]) + 1e-8))
        t = np.linspace(0, t_max, 20)
    
    spectrum = compute_laplacian_spectrum(points, k=min(len(points), 20))
    wave_trace = np.array([np.sum(np.cos(t_i * np.sqrt(np.abs(spectrum)))) for t_i in t])
    
    # Normalize the wave trace
    wave_trace = wave_trace / (np.linalg.norm(wave_trace) + 1e-8)
    return wave_trace

def wave_trace_distance(points1, points2):
    """Compute distance between wave trace signatures."""
    # Use same time points for both signatures
    spectrum1 = compute_laplacian_spectrum(points1, k=min(len(points1), 20))
    spectrum2 = compute_laplacian_spectrum(points2, k=min(len(points2), 20))
    
    t_max = min(
        2 * np.pi / (np.min(np.sqrt(np.abs(spectrum1[1:])) + 1e-8)),
        2 * np.pi / (np.min(np.sqrt(np.abs(spectrum2[1:])) + 1e-8))
    )
    t = np.linspace(0, t_max, 20)
    
    wt1 = wave_trace_signature(points1, t)
    wt2 = wave_trace_signature(points2, t)
    return np.linalg.norm(wt1 - wt2)


def main():
    # Select files using dialog
    file1, file2 = select_files()
    if file1 is None or file2 is None:
        return
    
    # Load points from JSON files
    points1 = load_points_from_json(file1)
    points2 = load_points_from_json(file2)
    
    # Calculate all metrics
    traditional_metrics = {
        "Hausdorff Distance": hausdorff_distance(points1, points2),
        "Earth Mover's Distance": earth_movers_distance(points1, points2),
        "Procrustes Distance": procrustes_distance(points1, points2),
        "Mutual Information": mutual_information_approximation(points1, points2),
        "Frechet Distance": frechet_distance_approximation(points1, points2)
    }
    
    spectral_metrics = {
        "Laplacian Spectral Distance": spectral_distance(points1, points2),
        "Heat Trace Distance": heat_trace_distance(points1, points2),
        "Wave Trace Distance": wave_trace_distance(points1, points2)
    }
    
    # Print results
    print(f"\nComparing spatial datasets:")
    print(f"File 1: {file1} ({len(points1)} points)")
    print(f"File 2: {file2} ({len(points2)} points)")
    
    print("\nTraditional Similarity Metrics:")
    print("-" * 50)
    for metric_name, value in traditional_metrics.items():
        print(f"{metric_name:25}: {value:.6f}")
    
    print("\nSpectral Similarity Metrics:")
    print("-" * 50)
    for metric_name, value in spectral_metrics.items():
        print(f"{metric_name:25}: {value:.6f}")

if __name__ == "__main__":
    main()