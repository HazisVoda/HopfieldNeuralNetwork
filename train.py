import numpy as np
from typing import Tuple, Optional
import pickle
import os

from hopfield import HopfieldNetwork
from data_utils import prepare_mnist_data, get_samples_per_digit, select_diverse_patterns, select_prototypical_patterns
import config


def train_hopfield_network(patterns: np.ndarray) -> HopfieldNetwork:
    n_neurons = patterns.shape[1]
    network = HopfieldNetwork(n_neurons)
    network.train(patterns)
    return network


def check_pattern_similarity(patterns: np.ndarray) -> dict:
    n_patterns = len(patterns)
    similarities = []
    
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            overlap = np.mean(patterns[i] == patterns[j])
            similarities.append(overlap)
    
    return {
        'mean_similarity': np.mean(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'num_high_similarity': sum(1 for s in similarities if s > 0.7)
    }


def prepare_training_data(
    crop_size: int = 16,
    num_patterns: int = 10,
    use_cropping: bool = True,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, int]:
    np.random.seed(random_seed)

    patterns, _, labels, image_size = prepare_mnist_data(
        split="train",
        digits=config.MNIST_DIGITS,
        num_samples=5000,
        random_seed=random_seed,
        crop_and_resize=use_cropping,
        crop_size=crop_size
    )
    
    # Calculate capacity
    n_neurons = image_size * image_size
    capacity = int(0.138 * n_neurons)
    
    print(f"\nCapacity Analysis:")
    print(f"  - Image size: {image_size}x{image_size} = {n_neurons} neurons")
    print(f"  - Theoretical capacity: ~{capacity} patterns")
    print(f"  - Requested patterns: {num_patterns}")
    
    if num_patterns > capacity:
        print(f"\nWARNING: Reducing to {capacity} patterns")
        num_patterns = capacity
    
    # Select prototypical patterns (one per digit)
    print(f"\nUsing prototypical pattern selection...")
    digits_to_use = config.MNIST_DIGITS[:num_patterns]
    patterns, labels = select_prototypical_patterns(patterns, labels, digits_to_use)
    
    # Check similarity - should be MUCH lower now with cropping
    sim_stats = check_pattern_similarity(patterns)
    print(f"\nPattern Similarity Analysis:")
    print(f"  - Mean similarity: {sim_stats['mean_similarity']:.2%}")
    print(f"  - Max similarity: {sim_stats['max_similarity']:.2%}")
    print(f"  - Highly similar pairs (>70%): {sim_stats['num_high_similarity']}")
    
    if sim_stats['mean_similarity'] < 0.7:
        print(f"Good! Pattern similarity is acceptable for Hopfield networks.")
    else:
        print(f"WARNING: High pattern similarity may cause issues.")
    
    print(f"\nPrepared {len(patterns)} training patterns")
    print(f"Digits included: {sorted(set(labels))}")
    
    return patterns, labels, image_size


def save_network(network: HopfieldNetwork, patterns: np.ndarray, 
                 labels: np.ndarray, image_size: int, 
                 filepath: str = "hopfield_model.pkl") -> None:
    data = {
        'network': network,
        'patterns': patterns,
        'labels': labels,
        'image_size': image_size
    }
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Model saved to {filepath}")


def load_network(filepath: str = "hopfield_model.pkl") -> Tuple[HopfieldNetwork, np.ndarray, np.ndarray, int]:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['network'], data['patterns'], data['labels'], data['image_size']


def main():
    print("\nHopfield Network Training")
    
    # Configuration
    crop_size = config.CROP_SIZE if config.USE_CROPPING else 28
    
    # Prepare data
    print("\n[1] Loading and preparing MNIST data...")
    patterns, labels, image_size = prepare_training_data(
        crop_size=crop_size,
        num_patterns=config.NUM_PATTERNS,
        use_cropping=config.USE_CROPPING,
        random_seed=config.RANDOM_SEED
    )
    
    # Train network
    print("\n[2] Training Hopfield network (Hebbian learning)...")
    network = train_hopfield_network(patterns)
    
    # Print network info
    info = network.get_storage_info()
    print(f"\nNetwork Statistics:")
    print(f"  - Neurons: {info['n_neurons']}")
    print(f"  - Patterns stored: {info['patterns_stored']}")
    print(f"  - Theoretical capacity: ~{info['theoretical_capacity']}")
    print(f"  - Load factor: {info['load_factor']:.2%}")
    
    # Save model
    print("\n[3] Saving trained model...")
    save_network(network, patterns, labels, image_size, "hopfield_model.pkl")
    print("\nTraining complete!")
    
    return network, patterns, labels, image_size


if __name__ == "__main__":
    main()