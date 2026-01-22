import argparse
import numpy as np
import os

import config
from hopfield import HopfieldNetwork
from data_utils import prepare_mnist_data, get_samples_per_digit
from noise import add_noise
from visualization import (
    plot_stored_patterns,
    plot_denoising_result,
    plot_multiple_denoising_results,
    plot_weight_matrix,
    setup_plotting_style
)
from train import train_hopfield_network, save_network, load_network, prepare_training_data
from evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hopfield Network Image Denoising",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--train', action='store_true',
        help='Train the network'
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Evaluate the network'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run interactive demo'
    )
    parser.add_argument(
        '--num-patterns', type=int, default=config.NUM_PATTERNS,
        help=f'Number of patterns to store (default: {config.NUM_PATTERNS})'
    )
    parser.add_argument(
        '--noise-level', type=float, default=config.NOISE_LEVEL,
        help=f'Noise level for testing (default: {config.NOISE_LEVEL})'
    )
    parser.add_argument(
        '--noise-type', type=str, default=config.NOISE_TYPE,
        choices=['flip', 'salt_pepper', 'mask', 'gaussian'],
        help=f'Type of noise (default: {config.NOISE_TYPE})'
    )
    parser.add_argument(
        '--crop-size', type=int, default=config.CROP_SIZE,
        help=f'Size after cropping digits (default: {config.CROP_SIZE})'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display plots (only save)'
    )
    
    return parser.parse_args()


def run_demo(network: HopfieldNetwork, patterns: np.ndarray, 
             labels: np.ndarray, image_size: int):
    print("\nInteractive Hopfield Network Demo")
    
    while True:
        print("\nOptions:")
        print("  1. Denoise a random pattern")
        print("  2. Denoise specific digit")
        print("  3. Test different noise levels")
        print("  4. View stored patterns")
        print("  5. View weight matrix")
        print("  6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
        except EOFError:
            break
        
        if choice == '1':
            # Random pattern
            idx = np.random.randint(len(patterns))
            pattern = patterns[idx]
            label = labels[idx]
            
            noise_level = float(input("Enter noise level (0.0-1.0, default 0.2): ") or 0.2)
            
            noisy = add_noise(pattern, "flip", noise_level, image_size)
            recovered, energy = network.recall(noisy, track_energy=True)
            
            accuracy = np.mean(pattern == recovered)
            print(f"\nDigit: {label}, Accuracy: {accuracy:.2%}")
            
            plot_denoising_result(
                pattern, noisy, recovered, image_size,
                title=f"Digit {label} - Noise: {noise_level:.0%}, Accuracy: {accuracy:.2%}"
            )
            
        elif choice == '2':
            digit = int(input("Enter digit (0-9): "))
            mask = labels == digit
            if not mask.any():
                print(f"No patterns for digit {digit}")
                continue
                
            digit_patterns = patterns[mask]
            idx = np.random.randint(len(digit_patterns))
            pattern = digit_patterns[idx]
            
            noise_level = float(input("Enter noise level (0.0-1.0, default 0.2): ") or 0.2)
            
            noisy = add_noise(pattern, "flip", noise_level, image_size)
            recovered, _ = network.recall(noisy)
            
            accuracy = np.mean(pattern == recovered)
            print(f"\nAccuracy: {accuracy:.2%}")
            
            plot_denoising_result(
                pattern, noisy, recovered, image_size,
                title=f"Digit {digit} - Accuracy: {accuracy:.2%}"
            )
            
        elif choice == '3':
            print("\nTesting noise levels: 0.1, 0.2, 0.3, 0.4, 0.5")
            
            idx = np.random.randint(len(patterns))
            pattern = patterns[idx]
            
            for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
                noisy = add_noise(pattern, "flip", noise_level, image_size)
                recovered, _ = network.recall(noisy)
                accuracy = np.mean(pattern == recovered)
                print(f"  Noise {noise_level:.0%}: Accuracy {accuracy:.2%}")
            
        elif choice == '4':
            plot_stored_patterns(patterns, image_size, labels)
            
        elif choice == '5':
            plot_weight_matrix(network.weights)
            
        elif choice == '6':
            break
        else:
            print("Invalid choice")


def main():
    args = parse_args()
    setup_plotting_style()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    show_plots = not args.no_show
    
    do_train = args.train or (not args.train and not args.evaluate and not args.demo)
    do_evaluate = args.evaluate or (not args.train and not args.evaluate and not args.demo)
    do_demo = args.demo
    
    network = None
    patterns = None
    labels = None
    image_size = None
    
    if do_train:
        print("\nTRAINING PHASE")
        print("\n1.Preparing training data...")
        
        patterns, labels, image_size = prepare_training_data(
            crop_size=config.CROP_SIZE,
            num_patterns=args.num_patterns,
            use_cropping=config.USE_CROPPING,
            random_seed=config.RANDOM_SEED
        )
        
        # Train
        print("\n2.Training Hopfield network...")
        network = train_hopfield_network(patterns)
        
        info = network.get_storage_info()
        print(f"\nNetwork Statistics:")
        print(f"  - Neurons: {info['n_neurons']}")
        print(f"  - Patterns stored: {info['patterns_stored']}")
        print(f"  - Theoretical capacity: ~{info['theoretical_capacity']}")
        print(f"  - Load factor: {info['load_factor']:.2%}")
        
        # Save
        print("\n3.Saving model...")
        save_network(network, patterns, labels, image_size)
        
        # Visualize stored patterns
        print("\n4.Visualizing stored patterns...")
        plot_stored_patterns(
            patterns[:min(50, len(patterns))], 
            image_size, 
            labels[:min(50, len(labels))],
            save_path=os.path.join(config.FIGURES_DIR, "stored_patterns.png"),
            show=show_plots
        )
    
    # Evaluation
    if do_evaluate:
        print("\nEVALUATION PHASE")
        
        # Load model if needed
        if network is None:
            print("\n1.Loading trained model...")
            try:
                network, patterns, labels, image_size = load_network()
                print(f"Loaded network with {network.patterns_stored} stored patterns")
            except FileNotFoundError:
                print("Error: No trained model found. Please run with --train first.")
                return
            
        print("\n2.Preparing test data...")
        print("  Note: Testing on stored patterns (Hopfield = associative memory)")
        
        test_patterns = patterns
        test_labels = labels
        
        # Run evaluation
        print("\n3.Running evaluation...")
        results = run_evaluation(
            network,
            test_patterns,
            test_labels,
            image_size,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
            num_samples=min(config.NUM_TEST_SAMPLES, len(test_patterns)),
            save_dir=config.FIGURES_DIR,
            show_plots=show_plots
        )
    
    # Demo mode
    if do_demo:
        if network is None:
            print("\n1.Loading trained model for demo...")
            try:
                network, patterns, labels, image_size = load_network()
            except FileNotFoundError:
                print("Error: No trained model found. Please run with --train first.")
                return
        
        run_demo(network, patterns, labels, image_size)

    print("\nComplete! Check the 'figures' directory for saved visualizations.")


if __name__ == "__main__":
    main()