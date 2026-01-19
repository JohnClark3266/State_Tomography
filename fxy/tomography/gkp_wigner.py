"""
GKP (Gottesman-Kitaev-Preskill) State Wigner Function Visualization

This script computes and plots the Wigner function of the ideal/approximate GKP state,
and outputs the Wigner function values at each sampling point.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def gkp_wigner(x: np.ndarray, p: np.ndarray, delta: float = 0.3, 
               n_peaks: int = 5) -> np.ndarray:
    """
    Compute the Wigner function of an approximate GKP state.
    
    The ideal GKP state is a grid of delta functions in phase space at positions
    (x, p) = (n*sqrt(pi), m*sqrt(pi)) for integers n, m.
    The approximate GKP state has Gaussian envelope with width parameter delta.
    
    W(x, p) ∝ Σ_{n,m} (-1)^{nm} * exp(-delta²/2 * ((x-n√π)² + (p-m√π)²)) 
                                * exp(-(x² + p²) * delta² / 2)
    
    Parameters:
    -----------
    x, p : np.ndarray
        Position and momentum coordinates (can be meshgrid)
    delta : float
        Squeezing parameter (smaller = more ideal, larger = more Gaussian envelope)
    n_peaks : int
        Number of peaks to include on each side of origin
    
    Returns:
    --------
    W : np.ndarray
        Wigner function values
    """
    sqrt_pi = np.sqrt(np.pi)
    W = np.zeros_like(x)
    
    # Sum over the grid of peaks
    for n in range(-n_peaks, n_peaks + 1):
        for m in range(-n_peaks, n_peaks + 1):
            # Position of this peak
            x0 = n * sqrt_pi
            p0 = m * sqrt_pi
            
            # Sign factor for GKP: (-1)^(n*m)
            sign = (-1) ** (n * m)
            
            # Gaussian peak at (x0, p0) with width from finite squeezing
            peak = np.exp(-((x - x0)**2 + (p - p0)**2) / (2 * delta**2))
            
            W += sign * peak
    
    # Gaussian envelope to simulate finite energy
    envelope = np.exp(-(x**2 + p**2) * delta**2 / 2)
    W = W * envelope
    
    # Normalize (approximate)
    W = W / (np.pi * delta**2)
    
    return W


def plot_gkp_wigner(x_range: Tuple[float, float] = (-6, 6),
                    p_range: Tuple[float, float] = (-6, 6),
                    n_points: int = 200,
                    delta: float = 0.3,
                    n_peaks: int = 5,
                    save_plot: bool = True,
                    save_data: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot the Wigner function of a GKP state and save the results.
    
    Parameters:
    -----------
    x_range, p_range : tuple
        Range of position and momentum coordinates
    n_points : int
        Number of sampling points in each dimension
    delta : float
        GKP state squeezing parameter
    n_peaks : int
        Number of peaks to include on each side
    save_plot : bool
        Whether to save the plot as an image
    save_data : bool
        Whether to save the Wigner function values to a file
    
    Returns:
    --------
    X, P, W : np.ndarray
        Meshgrid coordinates and Wigner function values
    """
    # Create sampling grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    X, P = np.meshgrid(x, p)
    
    # Compute Wigner function
    W = gkp_wigner(X, P, delta=delta, n_peaks=n_peaks)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use symmetric color scale around zero
    vmax = np.max(np.abs(W))
    vmin = -vmax
    
    # Contour plot
    cf = ax.contourf(X, P, W, levels=100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax, label='W(x, p)')
    
    # Mark sqrt(pi) grid lines
    sqrt_pi = np.sqrt(np.pi)
    for n in range(-3, 4):
        ax.axvline(n * sqrt_pi, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(n * sqrt_pi, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('Momentum p', fontsize=12)
    ax.set_title(f'GKP State Wigner Function (δ = {delta})', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = os.path.join(OUTPUT_DIR, f'gkp_wigner_delta_{delta}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
    
    plt.close()  # Close instead of show for non-interactive mode
    
    # Save Wigner function data
    if save_data:
        data_path = os.path.join(OUTPUT_DIR, f'gkp_wigner_data_delta_{delta}.npz')
        np.savez(data_path, x=x, p=p, X=X, P=P, W=W, delta=delta)
        print(f"✓ Data saved to: {data_path}")
        
        # Also save as CSV for easy inspection
        csv_path = os.path.join(OUTPUT_DIR, f'gkp_wigner_values_delta_{delta}.csv')
        with open(csv_path, 'w') as f:
            f.write("x,p,W(x,p)\n")
            for i in range(len(x)):
                for j in range(len(p)):
                    f.write(f"{X[j, i]:.6f},{P[j, i]:.6f},{W[j, i]:.10f}\n")
        print(f"✓ CSV saved to: {csv_path}")
    
    return X, P, W


def print_sample_values(X: np.ndarray, P: np.ndarray, W: np.ndarray, 
                        n_samples: int = 10):
    """
    Print some sample Wigner function values.
    
    Parameters:
    -----------
    X, P, W : np.ndarray
        Position, momentum coordinates and Wigner function values
    n_samples : int
        Number of sample points to print per region
    """
    print("\n" + "="*60)
    print("Sample Wigner Function Values")
    print("="*60)
    
    # Print values near the peaks (n*sqrt(pi), m*sqrt(pi))
    sqrt_pi = np.sqrt(np.pi)
    
    print("\n--- Values near lattice points (n√π, m√π) ---")
    for n in range(-2, 3):
        for m in range(-2, 3):
            x0, p0 = n * sqrt_pi, m * sqrt_pi
            # Find nearest grid point
            i = np.argmin(np.abs(X[0, :] - x0))
            j = np.argmin(np.abs(P[:, 0] - p0))
            print(f"  ({n:+d}√π, {m:+d}√π) = ({x0:+.3f}, {p0:+.3f}): "
                  f"W = {W[j, i]:+.6f}")
    
    print("\n--- Statistics ---")
    print(f"  Max W(x,p):  {np.max(W):.6f}")
    print(f"  Min W(x,p):  {np.min(W):.6f}")
    print(f"  Integral (approx): {np.sum(W) * (X[0,1]-X[0,0]) * (P[1,0]-P[0,0]):.6f}")
    

def main():
    """Main function to run the GKP Wigner function visualization."""
    print("="*60)
    print("GKP State Wigner Function Calculator")
    print("="*60)
    
    # Parameters
    delta = 0.3  # Squeezing parameter (smaller = more ideal)
    n_points = 200  # Sampling resolution
    x_range = (-6, 6)
    p_range = (-6, 6)
    
    print(f"\nParameters:")
    print(f"  δ (delta) = {delta}")
    print(f"  Sampling points: {n_points} x {n_points}")
    print(f"  x range: {x_range}")
    print(f"  p range: {p_range}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Compute and plot
    X, P, W = plot_gkp_wigner(
        x_range=x_range,
        p_range=p_range,
        n_points=n_points,
        delta=delta,
        save_plot=True,
        save_data=True
    )
    
    # Print sample values
    print_sample_values(X, P, W)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
