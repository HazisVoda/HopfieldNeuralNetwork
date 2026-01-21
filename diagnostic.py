import numpy as np
np.random.seed(42)
from matplotlib import pyplot as plt
import pandas as pd
import network
import os

def load_and_inspect_csv(filepath):
    """Load CSV and show detailed statistics"""
    print("="*60)
    print("DATA INSPECTION")
    print("="*60)
    
    df = pd.read_csv(filepath)
    
    if 'label' in df.columns:
        labels = df['label'].values
        pixels = df.drop('label', axis=1).values
    else:
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values
    
    print(f"Total samples: {len(labels)}")
    print(f"Pixels per image: {pixels.shape[1]}")
    print(f"Labels: {labels}")
    print(f"\nPixel value statistics:")
    print(f"  Min: {pixels.min()}")
    print(f"  Max: {pixels.max()}")
    print(f"  Mean: {pixels.mean():.2f}")
    print(f"  Std: {pixels.std():.2f}")
    
    return pixels, labels

def preprocessing_improved(pixels, threshold=0.5, verbose=False):
    """
    Improved preprocessing with better thresholding
    """
    # Normalize to [0, 1]
    if pixels.max() > 1:
        normalized = pixels / 255.0
    else:
        normalized = pixels
    
    # Apply threshold
    binary = normalized > threshold
    
    # Convert to {-1, 1}
    bipolar = 2 * binary.astype(int) - 1
    
    if verbose:
        print(f"  Pixels > {threshold}: {binary.sum()}/{len(binary)} ({100*binary.sum()/len(binary):.1f}%)")
        print(f"  Positive neurons: {(bipolar == 1).sum()}, Negative neurons: {(bipolar == -1).sum()}")
    
    return bipolar

def visualize_preprocessing(pixels, labels, threshold=0.5):
    """Show how preprocessing affects each pattern"""
    print("\n" + "="*60)
    print("PREPROCESSING VISUALIZATION")
    print("="*60)
    
    n_samples = min(5, len(pixels))
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original (normalized)
        if pixels[i].max() > 1:
            normalized = pixels[i] / 255.0
        else:
            normalized = pixels[i]
        img = normalized.reshape(28, 28)
        
        axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Digit {labels[i]} - Normalized')
        axes[i, 0].axis('off')
        
        # Binary
        binary = normalized > threshold
        axes[i, 1].imshow(binary.reshape(28, 28), cmap='gray')
        axes[i, 1].set_title(f'Binary (>{threshold})')
        axes[i, 1].axis('off')
        
        # Bipolar
        bipolar = 2 * binary.astype(int) - 1
        axes[i, 2].imshow(bipolar.reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
        axes[i, 2].set_title('Bipolar {-1,+1}')
        axes[i, 2].axis('off')
        
        # Print statistics
        pos_ratio = (bipolar == 1).sum() / len(bipolar)
        print(f"Digit {labels[i]}: {pos_ratio*100:.1f}% positive neurons")
    
    plt.tight_layout()
    plt.savefig('diagnostic_preprocessing.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved preprocessing visualization to 'diagnostic_preprocessing.png'")
    plt.close()

def check_pattern_similarity(data, labels):
    """Check how similar patterns are"""
    print("\n" + "="*60)
    print("PATTERN SIMILARITY ANALYSIS")
    print("="*60)
    
    n = len(data)
    similarities = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Calculate overlap (dot product normalized)
            overlap = np.dot(data[i], data[j]) / len(data[i])
            similarities[i, j] = overlap
    
    print("\nPattern overlap matrix:")
    print("(1.0 = identical, -1.0 = opposite, 0.0 = orthogonal)")
    print("\n     ", end="")
    for i in range(n):
        print(f"D{labels[i]:1d}  ", end="")
    print()
    
    for i in range(n):
        print(f"D{labels[i]:1d}  ", end="")
        for j in range(n):
            if i == j:
                print(" -- ", end="")
            else:
                print(f"{similarities[i,j]:+.2f}", end=" ")
        print()
    
    # Find most similar patterns
    print("\nMost similar pattern pairs:")
    max_sim = -2
    max_pair = None
    for i in range(n):
        for j in range(i+1, n):
            if similarities[i, j] > max_sim:
                max_sim = similarities[i, j]
                max_pair = (i, j)
    
    if max_pair:
        i, j = max_pair
        print(f"  Digit {labels[i]} and Digit {labels[j]}: overlap = {max_sim:.3f}")
        if max_sim > 0.3:
            print(f"  ⚠️  WARNING: These patterns are very similar! This can cause confusion.")

def test_with_fewer_patterns(pixels, labels, csv_filepath):
    """Test with progressively fewer patterns to find sweet spot"""
    print("\n" + "="*60)
    print("TESTING WITH FEWER PATTERNS")
    print("="*60)
    
    max_samples = len(labels)
    
    for n_patterns in [2, 3, 4, 5]:
        if n_patterns > max_samples:
            break
            
        print(f"\n--- Testing with {n_patterns} patterns ---")
        
        # Use first n patterns
        subset_pixels = pixels[:n_patterns]
        subset_labels = labels[:n_patterns]
        
        # Preprocess
        data = [preprocessing_improved(p, threshold=0.5) for p in subset_pixels]
        
        # Train
        model = network.HopfieldNetwork()
        model.train_weights(data)
        
        # Test with 30% noise
        test = [get_corrupted_input(d, 0.3) for d in data]
        predicted = model.predict(test, threshold=0, asyn=False)
        
        # Check accuracy
        accuracy = calculate_accuracy(data, predicted)
        print(f"  Accuracy: {accuracy:.2f}%")
        
        # Check if all predictions are the same (spurious state)
        all_same = all(np.array_equal(predicted[0], p) for p in predicted)
        if all_same:
            print(f"  ⚠️  WARNING: All predictions identical (spurious state)")
        else:
            print(f"  ✓ Good: Different patterns reconstructed differently")
            print(f"  → Recommendation: Use {n_patterns} patterns for best results")
            return n_patterns
    
    return 2  # Fallback

def get_corrupted_input(input, corruption_level):
    """Add noise by flipping pixels"""
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def calculate_accuracy(original, predicted):
    """Calculate pixel-wise accuracy"""
    total_correct = 0
    total_pixels = 0
    for orig, pred in zip(original, predicted):
        total_correct += np.sum(orig == pred)
        total_pixels += len(orig)
    return (total_correct / total_pixels) * 100

def reshape(data):
    """Reshape to 28x28"""
    return data.reshape(28, 28)

def run_diagnostic(csv_filepath):
    """Complete diagnostic of the Hopfield network"""
    print("\n" + "="*60)
    print("HOPFIELD NETWORK DIAGNOSTIC TOOL")
    print("="*60)
    
    # 1. Load and inspect data
    pixels, labels = load_and_inspect_csv(csv_filepath)
    
    # 2. Visualize preprocessing
    visualize_preprocessing(pixels, labels, threshold=0.5)
    
    # 3. Preprocess all patterns
    print("\n" + "="*60)
    print("PREPROCESSING ALL PATTERNS")
    print("="*60)
    data = []
    for i, p in enumerate(pixels):
        print(f"Pattern {i} (Digit {labels[i]}):")
        processed = preprocessing_improved(p, threshold=0.5, verbose=True)
        data.append(processed)
    
    # 4. Check pattern similarity
    check_pattern_similarity(data, labels)
    
    # 5. Test with different numbers of patterns
    recommended_n = test_with_fewer_patterns(pixels, labels, csv_filepath)
    
    # 6. Train with recommended number
    print("\n" + "="*60)
    print(f"FINAL TEST WITH {recommended_n} PATTERNS")
    print("="*60)
    
    subset_pixels = pixels[:recommended_n]
    subset_labels = labels[:recommended_n]
    data = [preprocessing_improved(p, threshold=0.5) for p in subset_pixels]
    
    model = network.HopfieldNetwork()
    model.train_weights(data)
    
    # Test reconstruction
    test = [get_corrupted_input(d, 0.3) for d in data]
    predicted = model.predict(test, threshold=0, asyn=False)
    
    # Visualize
    fig, axes = plt.subplots(recommended_n, 3, figsize=(9, 3*recommended_n))
    if recommended_n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(recommended_n):
        axes[i, 0].imshow(reshape(data[i]), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(f'Digit {subset_labels[i]}', fontweight='bold')
        
        axes[i, 1].imshow(reshape(test[i]), cmap='gray')
        axes[i, 1].set_title('Noisy (30%)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(reshape(predicted[i]), cmap='gray')
        axes[i, 2].set_title('Reconstructed')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnostic_result.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved reconstruction test to 'diagnostic_result.png'")
    plt.close()
    
    accuracy = calculate_accuracy(data, predicted)
    print(f"\nFinal accuracy: {accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  • diagnostic_preprocessing.png - Shows preprocessing steps")
    print("  • diagnostic_result.png - Shows final reconstruction")
    print(f"\n✓ Recommendation: Use {recommended_n} patterns for stable reconstruction")
    
    return recommended_n

if __name__ == '__main__':
    csv_filepath = "data/MNIST Digits.csv"
    
    if not os.path.exists(csv_filepath):
        print(f"ERROR: Cannot find '{csv_filepath}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    else:
        run_diagnostic(csv_filepath)