import numpy as np
np.random.seed(42)
from matplotlib import pyplot as plt
import pandas as pd
import network
import os

def get_corrupted_input(input, corruption_level):
    """Add noise by flipping pixels"""
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def reshape(data):
    """Reshape to 28x28"""
    return data.reshape(28, 28)

def preprocessing(pixels):
    """Preprocess to binary {-1, +1}"""
    normalized = pixels / 255.0
    binary = normalized > 0.5
    bipolar = 2 * binary.astype(int) - 1
    return bipolar

def load_mnist_csv(filepath, n_patterns=5):
    """Load ONLY the first n_patterns from CSV"""
    print(f"Loading MNIST data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Get only first n_patterns rows
    df = df.head(n_patterns)
    
    if 'label' in df.columns:
        labels = df['label'].values
        pixels = df.drop('label', axis=1).values
    else:
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values
    
    print(f"✓ Loaded {len(labels)} patterns")
    print(f"✓ Labels: {labels}")
    
    return pixels, labels

def plot_results(data, test, predicted, labels, noise_level, save_dir='results_fixed'):
    """Plot results"""
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    n_samples = len(data)
    fig, axarr = plt.subplots(n_samples, 3, figsize=(9, 2.5 * n_samples))
    
    if n_samples == 1:
        axarr = axarr.reshape(1, -1)
    
    for i in range(n_samples):
        if i == 0:
            axarr[i, 0].set_title('Original', fontsize=12, fontweight='bold')
            axarr[i, 1].set_title(f'Noisy ({int(noise_level*100)}%)', fontsize=12, fontweight='bold')
            axarr[i, 2].set_title('Reconstructed', fontsize=12, fontweight='bold')
        
        axarr[i, 0].imshow(data[i], cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 0].set_ylabel(f'Digit {labels[i]}', fontsize=11, fontweight='bold')
        
        axarr[i, 1].imshow(test[i], cmap='gray')
        axarr[i, 1].axis('off')
        
        axarr[i, 2].imshow(predicted[i], cmap='gray')
        axarr[i, 2].axis('off')
    
    plt.tight_layout()
    filename = f"{save_dir}/reconstruction_noise_{int(noise_level*100)}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {filename}")
    plt.close()

def calculate_accuracy(original, predicted):
    """Calculate accuracy"""
    total_correct = sum(np.sum(orig == pred) for orig, pred in zip(original, predicted))
    total_pixels = sum(len(orig) for orig in original)
    return (total_correct / total_pixels) * 100

def main():
    print("="*70)
    print("HOPFIELD NETWORK - FIXED VERSION (LIMITED PATTERNS)")
    print("="*70)
    
    csv_filepath = "data/MNIST Digits.csv"
    n_patterns = 5  # ← ONLY USE 5 PATTERNS!
    save_dir = 'results_fixed'
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # Load ONLY first 5 patterns
    print(f"\nLoading ONLY first {n_patterns} patterns...")
    pixels, labels = load_mnist_csv(csv_filepath, n_patterns=n_patterns)
    
    # Preprocess
    print("\nPreprocessing...")
    data = [preprocessing(p) for p in pixels]
    
    # Check balance
    print("\nPattern statistics:")
    for i, d in enumerate(data):
        pos_ratio = (d == 1).sum() / len(d)
        print(f"  Digit {labels[i]}: {pos_ratio*100:.1f}% active pixels")
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    model = network.HopfieldNetwork()
    model.train_weights(data)
    print(f"✓ Network trained with {len(data)} patterns")
    
    # Test with different noise levels
    print("\n" + "="*70)
    print("TESTING WITH DIFFERENT NOISE LEVELS")
    print("="*70)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for noise in noise_levels:
        print(f"\nNoise level: {int(noise*100)}%")
        test = [get_corrupted_input(d, noise) for d in data]
        predicted = model.predict(test, threshold=0, asyn=False)
        
        accuracy = calculate_accuracy(data, predicted)
        results.append(accuracy)
        print(f"  Accuracy: {accuracy:.2f}%")
        
        # Check if all predictions are the same (spurious state)
        all_same = all(np.array_equal(predicted[0], p) for p in predicted)
        if all_same:
            print(f"  ⚠️  WARNING: All predictions identical!")
        else:
            print(f"  ✓ Good: Different patterns reconstructed")
        
        # Save visualization
        plot_results(data, test, predicted, labels, noise, save_dir)
    
    # Plot noise robustness
    plt.figure(figsize=(10, 6))
    noise_percent = [n * 100 for n in noise_levels]
    plt.plot(noise_percent, results, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Noise Level (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Reconstruction Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'Noise Robustness ({n_patterns} patterns)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([50, 105])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/noise_robustness.png', dpi=150)
    print(f"\n✓ Saved noise robustness graph")
    plt.close()
    
    # Plot weight matrix
    model.plot_weights(save_dir)
    print(f"✓ Saved weight matrix")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\n✓ All results saved to '{save_dir}/' directory")
    print(f"\n📊 Summary:")
    print(f"  Patterns stored: {n_patterns}")
    print(f"  Best accuracy: {max(results):.2f}% at {noise_percent[results.index(max(results))]}% noise")
    print(f"  Worst accuracy: {min(results):.2f}% at {noise_percent[results.index(min(results))]}% noise")
    
    if min(results) > 90:
        print(f"\n✅ EXCELLENT: All accuracies above 90%!")
    elif min(results) > 80:
        print(f"\n✓ GOOD: Most accuracies above 80%")
    else:
        print(f"\n⚠️  Some accuracies low - try reducing to 3-4 patterns")

if __name__ == '__main__':
    main()