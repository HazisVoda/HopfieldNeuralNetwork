"""
Classic Hopfield Neural Network Demonstration with MNIST Dataset
================================================================

This script demonstrates the associative memory capabilities of a
classic Hopfield Network using the MNIST handwritten digits dataset.

Key concepts demonstrated:
1. Pattern storage in a Hopfield network
2. Pattern recall from noisy/corrupted inputs
3. Network capacity limitations
4. Convergence visualization

Author: Research Demo
Requirements: numpy, matplotlib, scikit-learn
Install: pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    
    def __init__(self):
        self.weights = None
        self.num_neurons = None
        self.patterns_stored = 0
    
    def train(self, patterns):
        patterns = np.array(patterns)
        self.patterns_stored = len(patterns)
        self.num_neurons = patterns.shape[1]
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        
        # Hebbian learning: sum of outer products
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        
        self.weights /= self.num_neurons
        
        np.fill_diagonal(self.weights, 0)
        
        # Calculate theoretical capacity
        capacity = 0.138 * self.num_neurons
        print(f"Network trained with {self.patterns_stored} patterns")
        print(f"Number of neurons: {self.num_neurons}")
        print(f"Theoretical capacity: ~{capacity:.0f} patterns")
        if self.patterns_stored > capacity:
            print("⚠️  Warning: Exceeding theoretical capacity may cause recall errors")
    
    def predict(self, pattern, max_iterations=100, track_energy=False):
        """
        Recall a pattern using asynchronous updates.
        
        Parameters:
        -----------
        pattern : np.ndarray
            Input pattern (potentially noisy)
        max_iterations : int
            Maximum number of update iterations
        track_energy : bool
            If True, return energy values at each step
            
        Returns:
        --------
        np.ndarray or tuple
            Recovered pattern, optionally with energy history
        """
        state = np.array(pattern).flatten().copy()
        
        if track_energy:
            energies = [self._compute_energy(state)]
        
        for iteration in range(max_iterations):
            previous_state = state.copy()
            
            # Asynchronous update: update neurons in random order
            update_order = np.random.permutation(self.num_neurons)
            
            for i in update_order:
                # Compute local field
                h = np.dot(self.weights[i], state)
                # Update neuron state
                state[i] = 1 if h >= 0 else -1
            
            if track_energy:
                energies.append(self._compute_energy(state))
            
            # Check for convergence
            if np.array_equal(state, previous_state):
                print(f"Converged after {iteration + 1} iterations")
                break
        else:
            print(f"Reached maximum iterations ({max_iterations})")
        
        if track_energy:
            return state, energies
        return state
    
    def predict_sync(self, pattern, max_iterations=100):
        """
        Recall a pattern using synchronous updates.
        All neurons update simultaneously.
        """
        state = np.array(pattern).flatten().copy()
        
        for iteration in range(max_iterations):
            previous_state = state.copy()
            
            # Synchronous update: all neurons at once
            h = np.dot(self.weights, state)
            state = np.where(h >= 0, 1, -1)
            
            if np.array_equal(state, previous_state):
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return state
    
    def _compute_energy(self, state):
        """
        Compute the energy of a state.
        E = -0.5 * Σ_ij w_ij * s_i * s_j
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))


def create_letter_pattern(letter, size=28):
    """
    Create distinct letter patterns that are more orthogonal.
    These patterns have minimal overlap for better Hopfield recall.
    """
    pattern = np.ones((size, size)) * -1  # Background
    
    if letter == 'A':
        # Triangle-like A
        for i in range(4, 24):
            # Two diagonal lines meeting at top
            left = 14 - (24-i)//2
            right = 14 + (24-i)//2
            if 8 <= left < 20:
                pattern[i, max(8,left):min(20,left+2)] = 1
            if 8 <= right < 20:
                pattern[i, max(8,right-1):min(20,right+1)] = 1
        pattern[14:17, 10:18] = 1  # Crossbar
        
    elif letter == 'H':
        # H shape
        pattern[4:24, 6:10] = 1   # Left vertical
        pattern[4:24, 18:22] = 1  # Right vertical
        pattern[12:16, 6:22] = 1  # Horizontal bar
        
    elif letter == 'X':
        # X shape with thick diagonals
        for i in range(28):
            j1 = i
            j2 = 27 - i
            if 4 <= i < 24:
                for di in range(-2, 3):
                    if 4 <= j1+di < 24:
                        pattern[i, j1+di] = 1
                    if 4 <= j2+di < 24:
                        pattern[i, j2+di] = 1
                        
    elif letter == 'O':
        # Thick O ring
        center = 14
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if 6 < dist < 11:
                    pattern[i, j] = 1
                    
    elif letter == 'T':
        # T shape
        pattern[4:8, 6:22] = 1    # Top horizontal
        pattern[4:24, 12:16] = 1  # Vertical stem
        
    elif letter == 'L':
        # L shape
        pattern[4:24, 6:10] = 1   # Vertical
        pattern[20:24, 6:22] = 1  # Horizontal base
        
    elif letter == 'I':
        # I shape
        pattern[4:8, 8:20] = 1    # Top serif
        pattern[4:24, 12:16] = 1  # Vertical
        pattern[20:24, 8:20] = 1  # Bottom serif
        
    elif letter == 'Z':
        # Z shape
        pattern[4:8, 6:22] = 1    # Top horizontal
        pattern[20:24, 6:22] = 1  # Bottom horizontal
        # Diagonal
        for i in range(4, 24):
            j = 22 - int((i-4) * 16 / 20)
            pattern[i, max(6,j-2):min(22,j+2)] = 1

    return pattern.flatten()


def create_digit_pattern(digit, size=28):
    """
    Create a simple binary representation of a digit.
    These are simplified digit patterns for demonstration.
    """
    pattern = np.ones((size, size)) * -1  # Background
    
    # Use letters for more distinct patterns
    letter_map = {0: 'O', 1: 'I', 2: 'Z', 3: 'H', 4: 'A', 5: 'T', 6: 'L', 7: 'X', 8: 'H', 9: 'A'}
    
    if digit in letter_map:
        return create_letter_pattern(letter_map[digit], size)
    
    return pattern.flatten()


def load_mnist_samples(patterns_to_use=['H', 'X', 'O'], samples_per_digit=1):
    """
    Create synthetic letter patterns for Hopfield network demonstration.
    
    Using letters instead of digits because they can be designed to be
    more orthogonal (less similar to each other), which is essential for
    classic Hopfield networks to work well.
    
    Parameters:
    -----------
    patterns_to_use : list
        Which letters to include (H, X, O, T, L, I, Z, A are available)
    samples_per_digit : int
        Number of samples per pattern (variations)
        
    Returns:
    --------
    tuple
        (patterns, labels) where patterns are binary +1/-1
    """
    print("Creating synthetic patterns...")
    
    patterns = []
    labels = []
    
    for letter in patterns_to_use:
        for sample in range(samples_per_digit):
            pattern = create_letter_pattern(letter)
            
            # Add slight variation for multiple samples
            if sample > 0:
                pattern_2d = pattern.reshape(28, 28)
                shift = np.random.randint(-1, 2)
                pattern_2d = np.roll(pattern_2d, shift, axis=0)
                pattern = pattern_2d.flatten()
            
            patterns.append(pattern)
            labels.append(letter)
    
    print(f"Created {len(patterns)} patterns: {patterns_to_use}")
    return np.array(patterns), np.array(labels)


def add_noise(pattern, noise_level=0.2):
    """
    Add noise to a pattern by flipping random bits.
    
    Parameters:
    -----------
    pattern : np.ndarray
        Original binary pattern
    noise_level : float
        Fraction of bits to flip (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Noisy pattern
    """
    noisy = pattern.copy()
    num_flips = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
    noisy[flip_indices] *= -1
    return noisy


def occlude_pattern(pattern, occlusion_type='bottom_half'):
    """
    Occlude part of a pattern to simulate partial information.
    
    Parameters:
    -----------
    pattern : np.ndarray
        Original pattern (784 elements for 28x28 image)
    occlusion_type : str
        Type of occlusion: 'bottom_half', 'top_half', 'left_half', 'right_half', 'random_half'
    """
    occluded = pattern.copy().reshape(28, 28)
    
    if occlusion_type == 'bottom_half':
        occluded[14:, :] = np.random.choice([-1, 1], size=(14, 28))
    elif occlusion_type == 'top_half':
        occluded[:14, :] = np.random.choice([-1, 1], size=(14, 28))
    elif occlusion_type == 'left_half':
        occluded[:, :14] = np.random.choice([-1, 1], size=(28, 14))
    elif occlusion_type == 'right_half':
        occluded[:, 14:] = np.random.choice([-1, 1], size=(28, 14))
    elif occlusion_type == 'random_half':
        mask = np.random.choice([True, False], size=(28, 28))
        occluded[mask] = np.random.choice([-1, 1], size=mask.sum())
    
    return occluded.flatten()


def visualize_recall(original, corrupted, recovered, title="Pattern Recall"):
    """Visualize the recall process."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original.reshape(28, 28), cmap='binary')
    axes[0].set_title('Original Pattern', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(corrupted.reshape(28, 28), cmap='binary')
    axes[1].set_title('Corrupted Input', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(recovered.reshape(28, 28), cmap='binary')
    axes[2].set_title('Recovered Pattern', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_all_patterns(patterns, labels, title="Stored Patterns"):
    """Visualize all stored patterns."""
    n = len(patterns)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    for i, (pattern, label) in enumerate(zip(patterns, labels)):
        axes[i].imshow(pattern.reshape(28, 28), cmap='binary')
        axes[i].set_title(f'Pattern: {label}', fontsize=11)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_energy_convergence(energies, title="Energy Convergence"):
    """Plot the energy function over iterations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energies, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def visualize_noise_robustness(network, pattern, label, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Test and visualize recall at different noise levels."""
    n = len(noise_levels) + 1
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    
    # Original
    axes[0, 0].imshow(pattern.reshape(28, 28), cmap='binary')
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for i, noise in enumerate(noise_levels, 1):
        noisy = add_noise(pattern, noise)
        recovered = network.predict(noisy)
        
        # Calculate accuracy
        accuracy = np.mean(recovered == pattern) * 100
        
        axes[0, i].imshow(noisy.reshape(28, 28), cmap='binary')
        axes[0, i].set_title(f'Noise: {noise*100:.0f}%', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recovered.reshape(28, 28), cmap='binary')
        axes[1, i].set_title(f'Recovered\n({accuracy:.1f}% match)', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle(f'Noise Robustness Test (Pattern {label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_occlusion_recovery(network, pattern, label):
    """Test pattern recovery from different occlusions."""
    occlusion_types = ['top_half', 'bottom_half', 'left_half', 'right_half']
    
    fig, axes = plt.subplots(2, len(occlusion_types) + 1, figsize=(15, 6))
    
    # Original
    axes[0, 0].imshow(pattern.reshape(28, 28), cmap='binary')
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].set_title('', fontsize=10)
    axes[1, 0].axis('off')
    
    for i, occ_type in enumerate(occlusion_types, 1):
        occluded = occlude_pattern(pattern, occ_type)
        recovered = network.predict(occluded)
        accuracy = np.mean(recovered == pattern) * 100
        
        axes[0, i].imshow(occluded.reshape(28, 28), cmap='binary')
        axes[0, i].set_title(occ_type.replace('_', ' ').title(), fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recovered.reshape(28, 28), cmap='binary')
        axes[1, i].set_title(f'Recovered\n({accuracy:.1f}% match)', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle(f'Occlusion Recovery Test (Pattern {label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def run_capacity_test(max_patterns=20):
    """
    Test network capacity by gradually increasing stored patterns.
    Uses random orthogonal-like patterns for accurate capacity measurement.
    """
    print("\n" + "="*60)
    print("CAPACITY TEST (Random Patterns)")
    print("="*60)
    
    # Generate random binary patterns (better for capacity testing)
    np.random.seed(123)  # For reproducibility
    all_patterns = np.random.choice([-1, 1], size=(max_patterns, 784))
    
    accuracies = []
    pattern_counts = list(range(1, max_patterns + 1))
    
    for n in pattern_counts:
        network = HopfieldNetwork()
        network.train(all_patterns[:n])
        
        # Test recall on all stored patterns with small noise
        correct = 0
        for pattern in all_patterns[:n]:
            noisy = add_noise(pattern, 0.05)  # Small noise
            recovered = network.predict(noisy)
            if np.array_equal(recovered, pattern):
                correct += 1
        
        accuracy = correct / n * 100
        accuracies.append(accuracy)
        print(f"Patterns: {n:2d}, Recall Accuracy: {accuracy:.1f}%")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pattern_counts, accuracies, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Perfect recall')
    theoretical_capacity = int(0.138 * 784)
    ax.axvline(x=theoretical_capacity, color='r', linestyle='--', alpha=0.5, 
               label=f'Theoretical capacity (~{theoretical_capacity})')
    ax.set_xlabel('Number of Stored Patterns', fontsize=12)
    ax.set_ylabel('Recall Accuracy (%)', fontsize=12)
    ax.set_title('Hopfield Network Capacity Test\n(Random Binary Patterns)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 110])
    ax.set_xlim([0, max_patterns + 1])
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run the complete Hopfield Network demonstration."""
    
    print("="*60)
    print("HOPFIELD NEURAL NETWORK - MNIST DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    PATTERNS_TO_USE = ['H', 'X', 'O']  # Distinct letter patterns
    SAMPLES_PER_PATTERN = 1           # Keep low for classic Hopfield
    NOISE_LEVEL = 0.15                # 15% of pixels flipped
    
    # -------------------------------------------------------------------------
    # 1. Load and prepare data
    # -------------------------------------------------------------------------
    print("\n[1] Creating patterns...")
    patterns, labels = load_mnist_samples(PATTERNS_TO_USE, SAMPLES_PER_PATTERN)
    
    # -------------------------------------------------------------------------
    # 2. Create and train the Hopfield network
    # -------------------------------------------------------------------------
    print("\n[2] Training Hopfield Network...")
    hopfield = HopfieldNetwork()
    hopfield.train(patterns)
    
    # -------------------------------------------------------------------------
    # 3. Visualize stored patterns
    # -------------------------------------------------------------------------
    print("\n[3] Visualizing stored patterns...")
    fig1 = visualize_all_patterns(patterns, labels, "Patterns Stored in Hopfield Network")
    plt.savefig('figures/01_stored_patterns.png', dpi=150, bbox_inches='tight')
    print("Saved: 01_stored_patterns.png")
    
    # -------------------------------------------------------------------------
    # 4. Demonstrate basic recall from noise
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating pattern recall from noisy input...")
    test_idx = 0
    original = patterns[test_idx]
    noisy = add_noise(original, NOISE_LEVEL)
    recovered, energies = hopfield.predict(noisy, track_energy=True)
    
    accuracy = np.mean(recovered == original) * 100
    print(f"Recovery accuracy: {accuracy:.1f}%")
    
    fig2 = visualize_recall(
        original, noisy, recovered,
        f"Pattern Recall Demo (Digit {labels[test_idx]}, {NOISE_LEVEL*100:.0f}% noise)"
    )
    plt.savefig('figures/02_basic_recall.png', dpi=150, bbox_inches='tight')
    print("Saved: 02_basic_recall.png")
    
    # -------------------------------------------------------------------------
    # 5. Visualize energy convergence
    # -------------------------------------------------------------------------
    print("\n[5] Visualizing energy convergence...")
    fig3 = visualize_energy_convergence(energies, "Energy Function During Pattern Recall")
    plt.savefig('figures/03_energy_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: 03_energy_convergence.png")
    
    # -------------------------------------------------------------------------
    # 6. Test robustness to different noise levels
    # -------------------------------------------------------------------------
    print("\n[6] Testing noise robustness...")
    fig4 = visualize_noise_robustness(
        hopfield, patterns[0], labels[0],
        noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    plt.savefig('figures/04_noise_robustness.png', dpi=150, bbox_inches='tight')
    print("Saved: 04_noise_robustness.png")
    
    # -------------------------------------------------------------------------
    # 7. Test occlusion recovery
    # -------------------------------------------------------------------------
    print("\n[7] Testing occlusion recovery...")
    fig5 = visualize_occlusion_recovery(hopfield, patterns[0], labels[0])
    plt.savefig('figures/05_occlusion_recovery.png', dpi=150, bbox_inches='tight')
    print("Saved: 05_occlusion_recovery.png")
    
    # -------------------------------------------------------------------------
    # 8. Test all stored patterns
    # -------------------------------------------------------------------------
    print("\n[8] Testing recall on all stored patterns...")
    fig, axes = plt.subplots(len(patterns), 3, figsize=(10, 3*len(patterns)))
    
    for i, (pattern, label) in enumerate(zip(patterns, labels)):
        noisy = add_noise(pattern, NOISE_LEVEL)
        recovered = hopfield.predict(noisy)
        accuracy = np.mean(recovered == pattern) * 100
        
        axes[i, 0].imshow(pattern.reshape(28, 28), cmap='binary')
        axes[i, 0].set_title(f'Original ({label})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy.reshape(28, 28), cmap='binary')
        axes[i, 1].set_title(f'Noisy ({NOISE_LEVEL*100:.0f}%)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(recovered.reshape(28, 28), cmap='binary')
        axes[i, 2].set_title(f'Recovered ({accuracy:.1f}%)')
        axes[i, 2].axis('off')
    
    plt.suptitle('Recall Test on All Stored Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/06_all_patterns_test.png', dpi=150, bbox_inches='tight')
    print("Saved: 06_all_patterns_test.png")
    
    # -------------------------------------------------------------------------
    # 9. Capacity test (optional - takes longer)
    # -------------------------------------------------------------------------
    print("\n[9] Running capacity test...")
    fig7 = run_capacity_test(max_patterns=20)
    plt.savefig('figures/07_capacity_test.png', dpi=150, bbox_inches='tight')
    print("Saved: 07_capacity_test.png")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  01_stored_patterns.png    - Patterns stored in the network")
    print("  02_basic_recall.png       - Basic recall demonstration")
    print("  03_energy_convergence.png - Energy function during recall")
    print("  04_noise_robustness.png   - Robustness to different noise levels")
    print("  05_occlusion_recovery.png - Recovery from partial occlusion")
    print("  06_all_patterns_test.png  - Recall test on all patterns")
    print("  07_capacity_test.png      - Network capacity analysis")
    print("\nYou can also display plots interactively by uncommenting plt.show()")
    
    # Uncomment to display plots interactively
    # plt.show()


if __name__ == "__main__":
    main()