import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os


def setup_plotting_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_pattern(pattern: np.ndarray, image_size: int, ax: Optional[plt.Axes] = None,
                 title: str = "", cmap: str = 'gray') -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    image = pattern.reshape(image_size, image_size)
    ax.imshow(image, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def plot_denoising_result(original: np.ndarray, noisy: np.ndarray, 
                          recovered: np.ndarray, image_size: int,
                          title: str = "", save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    noise_ratio = np.mean(original != noisy)
    recovery_accuracy = np.mean(original == recovered)

    plot_pattern(original, image_size, axes[0], "Original")
    plot_pattern(noisy, image_size, axes[1], f"Noisy ({noise_ratio:.1%} corrupted)")
    plot_pattern(recovered, image_size, axes[2], f"Recovered ({recovery_accuracy:.1%} accurate)")
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_multiple_denoising_results(originals: np.ndarray, noisys: np.ndarray,
                                     recovereds: np.ndarray, image_size: int,
                                     labels: Optional[np.ndarray] = None,
                                     n_samples: int = 5,
                                     save_path: Optional[str] = None,
                                     show: bool = True) -> plt.Figure:

    n_samples = min(n_samples, len(originals))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Column titles
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Noisy", fontsize=12, fontweight='bold')
    axes[0, 2].set_title("Recovered", fontsize=12, fontweight='bold')
    
    for i in range(n_samples):
        # Plot images
        axes[i, 0].imshow(originals[i].reshape(image_size, image_size), 
                          cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].imshow(noisys[i].reshape(image_size, image_size), 
                          cmap='gray', vmin=-1, vmax=1)
        axes[i, 2].imshow(recovereds[i].reshape(image_size, image_size), 
                          cmap='gray', vmin=-1, vmax=1)
        
        # Calculate accuracy
        accuracy = np.mean(originals[i] == recovereds[i])
        
        # Add label if available
        if labels is not None:
            axes[i, 0].set_ylabel(f"Digit: {labels[i]}\nAcc: {accuracy:.1%}", 
                                   fontsize=10)
        else:
            axes[i, 0].set_ylabel(f"Acc: {accuracy:.1%}", fontsize=10)
        
        for j in range(3):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_energy_convergence(energy_history: List[float], 
                            title: str = "Energy Convergence",
                            save_path: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    iterations = range(len(energy_history))
    ax.plot(iterations, energy_history, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Mark start and end
    ax.scatter([0], [energy_history[0]], color='green', s=100, zorder=5, label='Start')
    ax.scatter([len(energy_history)-1], [energy_history[-1]], color='red', s=100, zorder=5, label='End')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_accuracy_vs_noise(noise_levels: List[float], accuracies: List[float],
                           title: str = "Recovery Accuracy vs Noise Level",
                           save_path: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(noise_levels, accuracies, 'b-', linewidth=2, marker='s', markersize=8)
    ax.fill_between(noise_levels, accuracies, alpha=0.3)
    
    ax.set_xlabel("Noise Level (fraction of bits flipped)")
    ax.set_ylabel("Recovery Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(noise_levels))
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_stored_patterns(patterns: np.ndarray, image_size: int,
                         labels: Optional[np.ndarray] = None,
                         n_cols: int = 10, 
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
    n_patterns = len(patterns)
    n_rows = (n_patterns + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.atleast_2d(axes)
    
    for i in range(n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        if i < n_patterns:
            ax.imshow(patterns[i].reshape(image_size, image_size), 
                     cmap='gray', vmin=-1, vmax=1)
            if labels is not None:
                ax.set_title(f"{labels[i]}", fontsize=8)
        
        ax.axis('off')
    
    plt.suptitle(f"Stored Patterns (n={n_patterns})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_weight_matrix(weights: np.ndarray, 
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(weights, cmap='RdBu', aspect='equal')
    plt.colorbar(im, ax=ax, label='Weight')
    
    ax.set_title("Hopfield Network Weight Matrix")
    ax.set_xlabel("Neuron j")
    ax.set_ylabel("Neuron i")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig